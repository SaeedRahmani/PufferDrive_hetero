"""
Behavior Analysis Script
========================
Runs rollouts for one project across multiple scenarios and collects
comprehensive behavioral metrics:

  - Speed distribution (per agent, per scenario)
  - Acceleration / deceleration distribution
  - Max/min speed, max accel, max decel per agent
  - Jerk (rate of acceleration change) for smoothness
  - Goal achievement rate  (geometric: within 2 m of GT endpoint)
  - Collision/off-road incident rate  (reward-based: reward < -0.3)
  - ADE  (average displacement error vs ground-truth)
  - FDE  (final displacement error)
  - Fraction of active time per agent (vs full episode)

Usage (run from inside the project dir with its own venv):
    cd /data/lynx/saeani/pufferdrive_hetero_v2
    .venv/bin/python /users/saeani/src/guided_original/analyze_behavior.py \
        --checkpoint experiments/puffer_drive_eodvv1ay.pt \
        --run_name hetero_v2 \
        --output_dir /users/saeani/src/guided_original/behavior_results \
        --train_scenarios 0,1000,2000,3000,4000,5000,6000,7000,8000,8900 \
        --val_scenarios 0,100,200,300,400,500,600,700,800,900 \
        --dt 0.1
"""

import argparse
import configparser
import json
import os
import sys
import tempfile
import types

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",       required=True)
    p.add_argument("--run_name",         required=True)
    p.add_argument("--output_dir",       required=True)
    p.add_argument("--train_scenarios",
                   default="0,1000,2000,3000,4000,5000,6000,7000,8000,8900")
    p.add_argument("--val_scenarios",
                   default="0,100,200,300,400,500,600,700,800,900")
    p.add_argument("--episode_steps",    type=int, default=91)
    p.add_argument("--dt",               type=float, default=0.1,
                   help="Simulation timestep in seconds (default 0.1 = 10 Hz)")
    p.add_argument("--goal_dist_m",      type=float, default=4.0,
                   help="Distance threshold (m) for goal-achievement detection")
    p.add_argument("--incident_reward_thresh", type=float, default=-0.45,
                   help="Reward threshold below which an incident is flagged "
                        "(-0.45 catches collision=-0.5 events; guidance noise rarely below -0.4)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Architecture inference (shared with rollout_template.py)
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
        style_z_dim = 0
        style_enc_hidden = 0
    else:
        arch_type = "standard"
        style_z_dim = 0
        style_enc_hidden = 0

    return dict(
        arch_type       = arch_type,
        ego_features    = ego_feat,
        input_size      = sd["policy.ego_encoder.0.weight"].shape[0],
        hidden_size     = sd["policy.shared_embedding.1.weight"].shape[0],
        partner_features= sd["policy.partner_encoder.0.weight"].shape[1],
        road_features   = sd["policy.road_encoder.0.weight"].shape[1] - 6,
        lstm_hidden     = sd["lstm.weight_hh_l0"].shape[0] // 4,
        lstm_input      = sd["lstm.weight_ih_l0"].shape[1],
        style_z_dim     = style_z_dim,
        style_enc_hidden= style_enc_hidden,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Read env config from drive.ini
# ─────────────────────────────────────────────────────────────────────────────
def read_drive_config():
    cfg = configparser.ConfigParser(allow_no_value=True, strict=False,
                                    inline_comment_prefixes=(";", "#"))
    # Try both possible config locations (project root and inside pufferlib)
    cfg.read(["config/ocean/drive.ini", "pufferlib/config/ocean/drive.ini"])
    env_sec = cfg["env"] if "env" in cfg else {}

    def getf(key, default):
        v = env_sec.get(key, None)
        if v is None:
            return default
        v = v.split(";")[0].split("#")[0].strip().strip('"').strip("'")
        try:
            return float(v)
        except ValueError:
            return v

    def getb(key, default):
        v = env_sec.get(key, None)
        if v is None:
            return default
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


# ─────────────────────────────────────────────────────────────────────────────
# Build single-scenario env
# ─────────────────────────────────────────────────────────────────────────────
def build_env_for_scenario(Drive, map_file_path, arch, cfg, episode_steps):
    tmpdir = tempfile.mkdtemp(prefix="beh_map_")
    os.symlink(os.path.abspath(map_file_path),
               os.path.join(tmpdir, "map_000.bin"))

    kwargs = dict(
        map_dir                   = tmpdir,
        num_maps                  = 1,
        num_agents                = 128,
        episode_length            = episode_steps,
        init_steps                = 0,
        resample_frequency        = 99999,
        use_guided_autonomy       = int(cfg["use_guided_autonomy"]),
        use_guidance_observations = int(cfg["use_guidance_observations"]),
        guidance_speed_weight     = cfg["guidance_speed_weight"],
        guidance_heading_weight   = cfg["guidance_heading_weight"],
        prep_human_data           = False,
        save_data_to_disk         = False,
        reward_vehicle_collision  = cfg["reward_vehicle_collision"],
        reward_offroad_collision  = cfg["reward_offroad_collision"],
    )
    if cfg["guidance_dropout_prob"] is not None:
        kwargs["guidance_dropout_prob"] = cfg["guidance_dropout_prob"]
        kwargs["guidance_dropout_mode"] = cfg["guidance_dropout_mode"] or "max"
    if cfg["style_rand_prob"] is not None:
        kwargs["style_rand_prob"] = 0.0
    if arch["style_z_dim"] > 0:
        kwargs["style_z_dim"] = arch["style_z_dim"]

    try:
        return Drive(**kwargs), tmpdir
    except TypeError:
        for key in ("style_z_dim", "style_rand_prob"):
            kwargs.pop(key, None)
        return Drive(**kwargs), tmpdir


# ─────────────────────────────────────────────────────────────────────────────
# Build policy
# ─────────────────────────────────────────────────────────────────────────────
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
        kwargs["style_z_dim"] = arch["style_z_dim"]
        kwargs["style_encoder_hidden_size"] = arch["style_enc_hidden"]
        kwargs["style_stop_grad_rl"] = True

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
# Rollout with reward/termination capture
# ─────────────────────────────────────────────────────────────────────────────
def run_rollout_with_metrics(env, policy, device, episode_steps):
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

    frame_states   = [w0_state()]
    rewards_trace  = []   # list of (n_w0,) arrays
    term_trace     = []   # list of (n_w0,) bool arrays

    n_w0 = w0_end - w0_start

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

        obs_np, rew_np, term_np, trunc_np, _ = env.step(actions)
        rewards_trace.append(rew_np[w0_start:w0_end].copy())
        term_trace.append((term_np[w0_start:w0_end] |
                           trunc_np[w0_start:w0_end]).copy())
        frame_states.append(w0_state())

    return frame_states, rewards_trace, term_trace, n_w0


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation for one scenario
# ─────────────────────────────────────────────────────────────────────────────
def compute_scenario_metrics(frame_states, rewards_trace, gt_traj, n_w0,
                              dt, goal_dist_m, incident_thresh,
                              term_trace=None):
    """
    Returns a dict with per-agent arrays and aggregate scalars for one scenario.

    Key design choices
    ------------------
    * **Alive masking**: after an agent terminates (collision, off-road, or
      episode truncation), the env auto-resets it.  ``get_global_agent_state()``
      then reports the *reset* position, which is the agent's spawn point in a
      new episode.  We build an ``alive_mask`` from ``term_trace`` so that all
      position-based metrics (ADE, FDE, goal, kinematics) ignore post-reset
      frames.
    * Kinematic metrics (speed, accel, jerk) are computed **only for steps where
      the agent is alive AND has valid GT**.
    * Goal rate: geometric — agent gets within ``goal_dist_m`` of the last valid
      GT position at any point while the agent is alive.
    * Incident rate (reward-based): checks for steps where reward falls below
      ``incident_thresh`` **AND** the agent is alive.
    """
    T = len(frame_states)

    # Positions over time: shape (n_w0, T)
    xs = np.stack([s["x"][:n_w0] for s in frame_states], axis=1)   # (n_w0, T)
    ys = np.stack([s["y"][:n_w0] for s in frame_states], axis=1)

    # Rewards over time: shape (n_w0, T-1)
    rew_arr = np.stack(rewards_trace, axis=1).astype(np.float32)    # (n_w0, T-1)

    # ── Alive mask (exclude post-reset frames) ───────────────────────────────
    # term_trace[t] is a bool array of shape (n_w0,) — True if agent terminated
    # at action step t.  frame_states has T = len(term_trace) + 1 entries:
    #   frame_states[0]   = state after env.reset()
    #   frame_states[t+1] = state after env.step() at iteration t
    # When term_trace[t] is True, frame_states[t+1] is the auto-reset state
    # and is invalid for trajectory analysis.
    alive_mask = np.ones((n_w0, T), dtype=bool)
    if term_trace is not None:
        for i in range(n_w0):
            for t in range(len(term_trace)):
                if term_trace[t][i]:
                    alive_mask[i, t + 1:] = False
                    break

    # ── GT data and activity mask ─────────────────────────────────────────────
    gt_x     = gt_traj["x"][:n_w0]
    gt_y     = gt_traj["y"][:n_w0]
    gt_valid = gt_traj["valid"][:n_w0].astype(bool)

    # Squeeze out the num_maps=1 dimension that the vectorized env adds
    # Shape before: (n_w0, 1, T_gt); after: (n_w0, T_gt)
    if gt_x.ndim == 3:
        gt_x     = gt_x.squeeze(1)
        gt_y     = gt_y.squeeze(1)
        gt_valid = gt_valid.squeeze(1)

    T_gt = gt_valid.shape[1]

    # gt_active[i, t] = True iff agent i has valid GT at rollout step t AND is alive
    # (clipped to rollout length T, intersected with alive_mask)
    gt_active = np.zeros((n_w0, T), dtype=bool)
    clip_len  = min(T_gt, T)
    gt_active[:, :clip_len] = gt_valid[:, :clip_len]
    gt_active &= alive_mask

    # ── Masked kinematics ─────────────────────────────────────────────────────
    # Speed: valid only between two consecutive active steps AND physically
    # plausible.  Teleporting agents (gt_valid all-True but position jumps
    # from 0→actual) produce speed spikes >1000 m/s; 80 m/s is a generous
    # ceiling for any road vehicle in the Waymo dataset (~288 km/h).
    MAX_PLAUSIBLE_SPEED = 80.0   # m/s

    dx = np.diff(xs, axis=1)                        # (n_w0, T-1)
    dy = np.diff(ys, axis=1)
    raw_speed = np.sqrt(dx ** 2 + dy ** 2) / dt     # (n_w0, T-1)  [m/s]

    # Mask: both frame t and t+1 must be gt_active AND computed speed is sane.
    # The plausibility gate also excludes the step ADJACENT to a teleport
    # (which would otherwise show a huge accel/decel even if only one side
    # of the finite-difference is affected).
    speed_plausible = raw_speed <= MAX_PLAUSIBLE_SPEED
    speed_valid_mask = (gt_active[:, :-1] & gt_active[:, 1:]) & speed_plausible
    speed = np.where(speed_valid_mask, raw_speed, np.nan)      # NaN = excluded

    # Acceleration: valid only when BOTH bounding speed steps are valid AND
    # the computed accel is physically sane.  Startup transients (agents
    # going from rest → cruising speed in the first active step, e.g.
    # 0 → 25 m/s in 0.1s → 250 m/s²) are excluded by the 50 m/s² cap.
    MAX_PLAUSIBLE_ACCEL = 50.0  # m/s² (≈ 5g — generous upper bound)
    raw_acc  = np.diff(raw_speed, axis=1) / dt                 # (n_w0, T-2)
    acc_plausible = np.abs(raw_acc) <= MAX_PLAUSIBLE_ACCEL
    acc_valid = speed_valid_mask[:, :-1] & speed_valid_mask[:, 1:] & acc_plausible
    acc_arr   = np.where(acc_valid, raw_acc, np.nan)

    # Jerk
    raw_jerk = np.diff(raw_acc, axis=1) / dt                   # (n_w0, T-3)
    if raw_jerk.shape[1] > 0:
        jerk_valid = acc_valid[:, :-1] & acc_valid[:, 1:]
        jerk_arr   = np.where(jerk_valid, raw_jerk, np.nan)
    else:
        jerk_arr = np.full((n_w0, 0), np.nan)

    # ── Per-agent kinematic stats (ignoring NaN) ──────────────────────────────
    def nanstat(arr, func, axis=1):
        with np.errstate(all="ignore"):
            result = func(np.where(np.isfinite(arr), arr, np.nan), axis=axis)
        return result

    speed_mean   = np.nanmean(speed, axis=1)                     # (n_w0,)
    speed_max    = np.nanmax(speed,  axis=1)
    speed_p95    = np.nanpercentile(speed, 95, axis=1)
    max_accel    = np.nanmax(acc_arr,    axis=1)                 # max acceleration
    max_decel    = np.nanmax(-acc_arr,   axis=1)                 # max deceleration magnitude
    if jerk_arr.shape[1] > 0:
        mean_abs_jerk = np.nanmean(np.abs(jerk_arr), axis=1)
    else:
        mean_abs_jerk = np.full(n_w0, np.nan)

    # ── Goal achievement  (geometric: min dist to GT endpoint < goal_dist_m) ──
    goal_achieved = np.zeros(n_w0, dtype=bool)
    ade_values    = np.full(n_w0, np.nan)
    fde_values    = np.full(n_w0, np.nan)

    for i in range(n_w0):
        # Use gt_active (= gt_valid & alive_mask) so that post-reset frames
        # (where position = spawn point of new episode) are excluded.
        active_idx = np.where(gt_active[i])[0]
        if len(active_idx) == 0:
            continue

        # ADE: mean distance over all active steps
        dist = np.sqrt(
            (xs[i, active_idx] - gt_x[i, active_idx]) ** 2 +
            (ys[i, active_idx] - gt_y[i, active_idx]) ** 2
        )
        ade_values[i] = float(dist.mean())

        # FDE: distance at last active step (pre-termination, GT-valid)
        last_active = int(active_idx[-1])
        fde_values[i] = float(np.sqrt(
            (xs[i, last_active] - gt_x[i, last_active]) ** 2 +
            (ys[i, last_active] - gt_y[i, last_active]) ** 2
        ))

        # Goal: min distance to GT endpoint at any active step
        valid_mask = gt_valid[i]
        valid_idx = np.where(valid_mask)[0]
        last_gt_idx  = int(valid_idx[-1])
        goal_x_i     = float(gt_x[i, last_gt_idx])
        goal_y_i     = float(gt_y[i, last_gt_idx])
        active_xs = xs[i, active_idx]
        active_ys = ys[i, active_idx]
        min_dist = float(np.sqrt(
            (active_xs - goal_x_i) ** 2 + (active_ys - goal_y_i) ** 2
        ).min())
        goal_achieved[i] = min_dist < goal_dist_m

    # ── Incident detection (reward-based, active steps only) ──────────────────
    # Only flag incidents for steps where the agent is active (gt_active).
    # This avoids false positives from reward noise during inactive frames.
    # We use a threshold of -(0.45) which is just ABOVE the collision penalty
    # (-0.5), so it catches true collision/offroad events while filtering out
    # guidance-related noise (which rarely goes below -0.3).
    active_mask_for_rew = gt_active[:, :rew_arr.shape[1]]         # (n_w0, T-1)
    # Count active steps per agent
    active_steps_per_agent = active_mask_for_rew.sum(axis=1)      # (n_w0,)

    rew_active = np.where(active_mask_for_rew, rew_arr, 0.0)      # inactive → 0 (neutral)
    incident_per_agent = (
        (rew_arr < incident_thresh) & active_mask_for_rew
    ).any(axis=1)                                                  # (n_w0,)

    incident_steps_per_agent = (
        (rew_arr < incident_thresh) & active_mask_for_rew
    ).sum(axis=1)                                                  # (n_w0,)

    severe_incident_per_agent = (
        (rew_arr < -0.8) & active_mask_for_rew
    ).any(axis=1)

    # Mean reward per agent (only active steps)
    mean_reward_per_agent = np.where(
        active_steps_per_agent > 0,
        (rew_active * active_mask_for_rew).sum(axis=1) / np.maximum(active_steps_per_agent, 1),
        np.nan
    )

    # ── Aggregate over agents ──────────────────────────────────────────────────
    def safe_mean(arr):
        finite = arr[np.isfinite(arr)]
        return float(finite.mean()) if len(finite) > 0 else float("nan")

    metrics = {
        # Per-agent arrays (serialized as lists for JSON)
        "speed_mean_per_agent":        [float(x) for x in speed_mean],
        "speed_max_per_agent":         [float(x) for x in speed_max],
        "speed_p95_per_agent":         [float(x) for x in speed_p95],
        "max_accel_per_agent":         [float(x) for x in max_accel],
        "max_decel_per_agent":         [float(x) for x in max_decel],
        "mean_abs_jerk_per_agent":     [float(x) for x in mean_abs_jerk],
        "ade_per_agent":               ade_values.tolist(),
        "fde_per_agent":               fde_values.tolist(),
        "goal_achieved_per_agent":     goal_achieved.tolist(),
        "incident_per_agent":          incident_per_agent.tolist(),
        "incident_steps_per_agent":    incident_steps_per_agent.tolist(),
        "mean_reward_per_agent":       [float(x) for x in mean_reward_per_agent],
        # Scenario-level aggregates
        "n_agents":                    int(n_w0),
        "goal_rate":                   float(goal_achieved.mean()),
        "incident_rate":               float(incident_per_agent.mean()),
        "severe_incident_rate":        float(severe_incident_per_agent.mean()),
        "mean_speed":                  safe_mean(speed_mean),
        "mean_max_speed":              safe_mean(speed_max),
        "mean_speed_p95":              safe_mean(speed_p95),
        "mean_max_accel":              safe_mean(max_accel),
        "mean_max_decel":              safe_mean(max_decel),
        "mean_jerk":                   safe_mean(mean_abs_jerk),
        "ade":                         safe_mean(ade_values),
        "fde":                         safe_mean(fde_values),
        "incident_steps_mean":         float(incident_steps_per_agent.mean()),
        "mean_reward":                 safe_mean(mean_reward_per_agent),
    }

    # Store masked step-level samples for distribution plots (NaN excluded)
    speed_flat = speed.flatten()
    acc_flat   = acc_arr.flatten()
    jerk_flat  = jerk_arr.flatten() if jerk_arr.shape[1] > 0 else np.array([])

    metrics["speed_all_steps"] = [float(x) for x in speed_flat[np.isfinite(speed_flat)]]
    metrics["accel_all_steps"] = [float(x) for x in acc_flat[np.isfinite(acc_flat)]]
    metrics["jerk_all_steps"]  = [float(x) for x in jerk_flat[np.isfinite(jerk_flat)]]

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate across scenarios
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_project_metrics(all_scenario_metrics):
    """Pool per-agent arrays across all scenarios and compute overall stats."""
    keys_to_pool = [
        "speed_mean_per_agent", "speed_max_per_agent", "speed_p95_per_agent",
        "max_accel_per_agent", "max_decel_per_agent", "mean_abs_jerk_per_agent",
        "ade_per_agent", "fde_per_agent",
        "goal_achieved_per_agent", "incident_per_agent",
    ]
    # Heavy step-level samples → pool them too for CDFs
    step_keys = ["speed_all_steps", "accel_all_steps", "jerk_all_steps"]

    pooled = {k: [] for k in keys_to_pool}
    step_pooled = {k: [] for k in step_keys}

    for sc in all_scenario_metrics:
        for k in keys_to_pool:
            if k in sc:
                pooled[k].extend(sc[k])
        for k in step_keys:
            if k in sc:
                step_pooled[k].extend(sc[k])

    def stats(lst):
        a = np.asarray(lst, dtype=np.float32)
        finite = a[np.isfinite(a)]
        if len(finite) == 0:
            return {}
        return {
            "mean":  float(finite.mean()),
            "std":   float(finite.std()),
            "median": float(np.median(finite)),
            "p5":    float(np.percentile(finite, 5)),
            "p25":   float(np.percentile(finite, 25)),
            "p75":   float(np.percentile(finite, 75)),
            "p95":   float(np.percentile(finite, 95)),
            "max":   float(finite.max()),
            "min":   float(finite.min()),
        }

    agg = {}
    for k, lst in pooled.items():
        agg[k] = stats(lst)

    for k, lst in step_pooled.items():
        agg[k] = stats(lst)

    # Also compute simple aggregate rate metrics
    n_agents = sum(sc["n_agents"] for sc in all_scenario_metrics)
    goal_total    = sum(sum(sc.get("goal_achieved_per_agent", [])) for sc in all_scenario_metrics)
    incident_total= sum(sum(sc.get("incident_per_agent", [])) for sc in all_scenario_metrics)

    agg["total_agents"]    = n_agents
    agg["overall_goal_rate"]     = float(goal_total / n_agents) if n_agents > 0 else float("nan")
    agg["overall_incident_rate"] = float(incident_total / n_agents) if n_agents > 0 else float("nan")

    # Mean over scenarios for scalar metrics
    for key in ["goal_rate", "incident_rate", "ade", "fde",
                "mean_speed", "mean_max_speed", "mean_max_accel", "mean_max_decel",
                "mean_jerk", "mean_speed_p95"]:
        vals = [sc[key] for sc in all_scenario_metrics if np.isfinite(sc.get(key, float("nan")))]
        if vals:
            agg[f"scenario_mean_{key}"] = float(np.mean(vals))
            agg[f"scenario_std_{key}"]  = float(np.std(vals))

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
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
    print(f"DT        : {args.dt}s  |  goal_dist={args.goal_dist_m}m  |  "
          f"incident_thresh={args.incident_reward_thresh}")

    sd   = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd   = {k.replace("module.", ""): v for k, v in sd.items()}
    arch = infer_arch(sd)
    cfg  = read_drive_config()

    # Override style_z_dim from config if available (handles vae_1d_ll0
    # which infer_arch misclassifies as hetero with z=0 because
    # ego_features=191 matches both vae_1d (190+1) and hetero (190+1))
    if cfg["style_z_dim"] is not None and int(cfg["style_z_dim"]) != arch["style_z_dim"]:
        arch["style_z_dim"] = int(cfg["style_z_dim"])
        # Also fix arch_type: if we detected hetero but config says style_z > 0,
        # it's actually a vae model with a 1D latent
        if arch["arch_type"] == "hetero" and arch["style_z_dim"] > 0:
            arch["arch_type"] = "vae"

    print(f"Arch      : {arch['arch_type']}  (ego={arch['ego_features']}, "
          f"lstm={arch['lstm_hidden']}, style_z={arch['style_z_dim']})")

    map_subdirs = {
        "train": "pufferlib/resources/drive/binaries/training",
        "val":   "pufferlib/resources/drive/binaries/validation",
    }
    scenario_map = {
        "train": [int(x) for x in args.train_scenarios.split(",")],
        "val":   [int(x) for x in args.val_scenarios.split(",")],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    run_safe = args.run_name.replace("/", "_").replace(" ", "_")
    out_json = os.path.join(args.output_dir, f"{run_safe}_behavior.json")

    if os.path.exists(out_json):
        print(f"[SKIP] Already exists: {out_json}")
        return

    all_scenario_metrics = []
    scenario_index       = []

    for split, indices in scenario_map.items():
        print(f"\n── {split.upper()} scenarios: {indices}")
        map_dir  = map_subdirs[split]
        all_maps = sorted(
            [f for f in os.listdir(map_dir) if f.endswith(".bin")],
            key=lambda f: int(f.replace("map_", "").replace(".bin", ""))
        )

        for idx in indices:
            map_file = os.path.join(map_dir, all_maps[idx])
            sc_label = f"{split}_{all_maps[idx].replace('.bin','')}"
            print(f"  Scenario {sc_label}  ({map_file})")

            import shutil
            env, tmpdir = build_env_for_scenario(
                Drive, map_file, arch, cfg, args.episode_steps
            )
            try:
                policy = build_policy(env, arch, sd, device, ocean_torch, pufferlib)

                env.reset()
                gt = env.get_ground_truth_trajectories()

                env.reset()
                frame_states, rewards_trace, term_trace, n_w0 = \
                    run_rollout_with_metrics(env, policy, device, args.episode_steps)

                print(f"    n_w0={n_w0}, T={len(frame_states)}, "
                      f"rew_steps={len(rewards_trace)}")

                sc_metrics = compute_scenario_metrics(
                    frame_states, rewards_trace, gt, n_w0,
                    dt           = args.dt,
                    goal_dist_m  = args.goal_dist_m,
                    incident_thresh = args.incident_reward_thresh,
                    term_trace   = term_trace,
                )
                sc_metrics["scenario"] = sc_label
                sc_metrics["split"]    = split
                sc_metrics["map_idx"]  = idx

                all_scenario_metrics.append(sc_metrics)
                scenario_index.append(sc_label)

                # Print quick summary
                print(f"    goal_rate={sc_metrics['goal_rate']:.2f}  "
                      f"incident_rate={sc_metrics['incident_rate']:.2f}  "
                      f"mean_speed={sc_metrics['mean_speed']:.1f} m/s  "
                      f"ade={sc_metrics['ade']:.2f} m")

            finally:
                env.close()
                shutil.rmtree(tmpdir, ignore_errors=True)

    # Aggregate across all scenarios
    project_agg = aggregate_project_metrics(all_scenario_metrics)

    output = {
        "_run_name":   args.run_name,
        "_checkpoint": args.checkpoint,
        "_n_scenarios": len(all_scenario_metrics),
        "_episode_steps": args.episode_steps,
        "_dt": args.dt,
        "_goal_dist_m": args.goal_dist_m,
        "_incident_reward_thresh": args.incident_reward_thresh,
        "aggregate": project_agg,
        "scenarios": all_scenario_metrics,
    }

    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_json}")

    # Print summary
    ag = project_agg
    print(f"\n{'─'*60}")
    print(f"Project summary: {args.run_name}")
    print(f"  Scenarios : {len(all_scenario_metrics)}")
    print(f"  Goal rate : {ag.get('overall_goal_rate', float('nan')):.3f}")
    print(f"  Inc. rate : {ag.get('overall_incident_rate', float('nan')):.3f}")
    if "speed_mean_per_agent" in ag:
        sp = ag["speed_mean_per_agent"]
        print(f"  Speed     : mean={sp.get('mean',0):.1f}  "
              f"p95={sp.get('p95',0):.1f}  max={sp.get('max',0):.1f} m/s")
    if "max_accel_per_agent" in ag:
        ac = ag["max_accel_per_agent"]
        print(f"  Max accel : mean={ac.get('mean',0):.2f}  "
              f"p95={ac.get('p95',0):.2f} m/s²")
    if "max_decel_per_agent" in ag:
        dc = ag["max_decel_per_agent"]
        print(f"  Max decel : mean={dc.get('mean',0):.2f}  "
              f"p95={dc.get('p95',0):.2f} m/s²")
    if "ade_per_agent" in ag:
        ad = ag["ade_per_agent"]
        print(f"  ADE       : mean={ad.get('mean',0):.2f}  "
              f"median={ad.get('median',0):.2f} m")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
