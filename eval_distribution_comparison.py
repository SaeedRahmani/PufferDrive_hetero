"""
Distribution Comparison: Run 200 scenarios at z=-2 and z=+2 (PC1)
and save raw per-timestep acceleration and speed data for all agents.
"""

import argparse
import ast
import configparser
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-scenarios", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default=None)
    args_cli = parser.parse_args()

    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT = os.path.join(
        PROJECT_DIR, "experiments/puffer_drive_lpes16fi/model_puffer_drive_005801.pt"
    )
    Z_DATA_PATH = os.path.join(
        PROJECT_DIR, "pufferlib/resources/drive/human_demonstrations/expert_style_z_h32.pt"
    )

    target_scenarios = args_cli.target_scenarios
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args_cli.output_dir:
        output_dir = args_cli.output_dir
    else:
        output_dir = os.path.join(
            PROJECT_DIR, f"evaluation_results/dist_comparison_{timestamp}"
        )
    os.makedirs(output_dir, exist_ok=True)

    # ── Setup paths ──────────────────────────────────────────────────────────
    sys.path.insert(0, PROJECT_DIR)
    os.chdir(PROJECT_DIR)

    import torch
    import pufferlib
    import pufferlib.vector
    import pufferlib.pytorch
    from pufferlib.ocean.environment import env_creator
    import pufferlib.ocean.torch as ocean_torch

    # ── Compute PCA from expert z data ───────────────────────────────────────
    z_data = torch.load(Z_DATA_PATH, map_location="cpu", weights_only=True).numpy()
    z_centered = z_data - z_data.mean(axis=0)
    cov = np.cov(z_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    var_ratios = eigvals / eigvals.sum()
    pc1 = eigvecs[:, 0].astype(np.float32)
    pc1 /= np.linalg.norm(pc1)

    # Compute sigma (std of projections onto PC1)
    projections = z_centered @ pc1
    sigma = float(projections.std())

    print(f"PC1: {pc1}  var_ratio: {var_ratios[0]:.3f}  sigma: {sigma:.4f}")

    # z-scalars to evaluate: -2σ and +2σ
    z_scalars = {
        "z_neg2": -2.0 * sigma,
        "z_pos2": +2.0 * sigma,
    }
    print(f"Sweep z-scalars: { {k: f'{v:.4f}' for k, v in z_scalars.items()} }")

    # ── Load config ──────────────────────────────────────────────────────────
    puffer_dir = os.path.dirname(os.path.realpath(pufferlib.__file__))
    default_config = os.path.join(puffer_dir, "config", "default.ini")
    drive_config = os.path.join(puffer_dir, "config", "ocean", "drive.ini")

    p = configparser.ConfigParser()
    p.read([default_config, drive_config])

    def puffer_type(value):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value

    cfg = defaultdict(dict)
    for section in p.sections():
        for key in p[section]:
            if section == "base":
                cfg[key] = puffer_type(p[section][key])
            else:
                cfg[section][key] = puffer_type(p[section][key])

    cfg["train"]["use_rnn"] = cfg.get("rnn_name") is not None

    cfg["env"]["prep_human_data"] = False
    cfg["env"]["termination_mode"] = 0
    cfg["eval"]["wosac_realism_eval"] = True
    cfg["eval"]["wosac_target_scenarios"] = target_scenarios
    cfg["env"]["map_dir"] = cfg["eval"]["map_dir"]
    cfg["vec"] = dict(backend="PufferEnv", num_envs=1)
    cfg["env"]["init_mode"] = cfg["eval"]["wosac_init_mode"]
    cfg["env"]["control_mode"] = cfg["eval"]["wosac_control_mode"]
    cfg["env"]["init_steps"] = cfg["eval"]["wosac_init_steps"]
    cfg["env"]["goal_behavior"] = cfg["eval"]["wosac_goal_behavior"]
    cfg["env"]["goal_radius"] = cfg["eval"]["wosac_goal_radius"]

    num_scenes_per_batch = cfg["eval"]["wosac_batch_size"]
    cfg["env"]["num_agents"] = num_scenes_per_batch * 10
    cfg["env"]["num_maps"] = cfg["eval"]["wosac_scenario_pool_size"]
    cfg["load_model_path"] = CHECKPOINT
    cfg["load_id"] = None

    # ── Create environment ───────────────────────────────────────────────────
    make_env = env_creator("puffer_drive")
    vecenv = pufferlib.vector.make(make_env, env_kwargs=cfg["env"], **cfg["vec"])

    device = cfg["train"]["device"]
    policy_cls = getattr(ocean_torch, cfg["policy_name"])
    policy = policy_cls(vecenv.driver_env, **cfg.get("policy", {}))

    rnn_name = cfg.get("rnn_name")
    if rnn_name is not None:
        rnn_cls = getattr(ocean_torch, rnn_name)
        policy = rnn_cls(vecenv.driver_env, policy, **cfg.get("rnn", {}))

    policy = policy.to(device)
    state_dict = torch.load(CHECKPOINT, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)
    policy.eval()
    print(f"Policy loaded: {sum(p.numel() for p in policy.parameters())} params")

    style_z_dim = int(cfg["env"].get("style_z_dim", 4))
    num_steps = 91
    init_steps = cfg.get("eval", {}).get("wosac_init_steps", 0)
    sim_steps = num_steps - init_steps
    num_rollouts = cfg.get("eval", {}).get("wosac_num_rollouts", 32)
    max_batches = cfg["eval"].get("wosac_max_batches", 100)

    # ── Helper: run rollouts for one z value ─────────────────────────────────
    def run_for_z(z_label, z_scalar):
        """Run target_scenarios with given z_scalar along PC1.
        Returns dict with raw per-timestep speed and acceleration arrays.
        """
        z_vec = pc1 * z_scalar  # 4D z vector along PC1 direction

        all_speeds = []      # all valid speed samples (flat)
        all_accels = []      # all valid accel samples (flat)
        all_per_agent_p85 = []  # p85 speed per agent

        num_agents = vecenv.observation_space.shape[0]
        unique_files = set()
        batch_idx = 0

        while batch_idx < max_batches and len(unique_files) < target_scenarios:
            if batch_idx > 0:
                vecenv.driver_env.resample_maps()

            # Track scenario IDs
            gt = vecenv.get_ground_truth_trajectories()
            unique_files.update(str(s) for s in np.unique(gt["scenario_id"]))

            for rollout_idx in range(num_rollouts):
                # Set style z
                z_array = np.tile(z_vec, (num_agents, 1)).astype(np.float32)
                vecenv.driver_env.set_style_z(z_array)

                obs, info = vecenv.reset()
                truncations = np.zeros((num_agents,), dtype=bool)
                state = {}
                if cfg["train"]["use_rnn"] and policy is not None:
                    state = dict(
                        lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                        lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
                    )

                agent_done = np.zeros(num_agents, dtype=bool)
                rollout_speeds = []  # (T, num_agents)
                dt = 0.1
                MAX_SPEED = 100.0  # from drive.h

                for time_idx in range(sim_steps):
                    agent_done |= truncations

                    # Speed directly from observation: obs[:,2] = signed_speed / MAX_SPEED
                    speed = np.abs(obs[:, 2]) * MAX_SPEED
                    speed = np.where(~agent_done, speed, np.nan)
                    rollout_speeds.append(speed)
                    all_speeds.append(speed)

                    if len(rollout_speeds) >= 2:
                        prev_speed = rollout_speeds[-2]
                        accel = (speed - prev_speed) / dt
                        all_accels.append(accel)

                    with torch.no_grad():
                        ob_tensor = torch.as_tensor(obs).to(device)
                        logits, value = policy.forward_eval(ob_tensor, state)
                        action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                        action_np = action.cpu().numpy().reshape(vecenv.action_space.shape)

                    obs, rewards, terminals, truncations, infos = vecenv.step(action_np)

                # Per-agent p85 speed for this rollout
                if rollout_speeds:
                    speed_arr = np.stack(rollout_speeds, axis=0)  # (T, num_agents)
                    for agent_idx in range(num_agents):
                        agent_speeds = speed_arr[:, agent_idx]
                        agent_speeds = agent_speeds[~np.isnan(agent_speeds)]
                        if len(agent_speeds) > 5:
                            all_per_agent_p85.append(np.percentile(agent_speeds, 85))

            batch_idx += 1
            print(f"  [{z_label}] Batch {batch_idx}: "
                  f"{len(unique_files)}/{target_scenarios} scenarios, "
                  f"{len(all_speeds)} speed samples")

        # Flatten
        speed_flat = np.concatenate([s.reshape(-1) for s in all_speeds])
        speed_flat = speed_flat[~np.isnan(speed_flat)]
        accel_flat = np.concatenate([a.reshape(-1) for a in all_accels])
        accel_flat = accel_flat[~np.isnan(accel_flat)]
        p85_arr = np.array(all_per_agent_p85)

        print(f"  [{z_label}] Done: {len(speed_flat)} speed, "
              f"{len(accel_flat)} accel, {len(p85_arr)} p85 samples")

        return {
            "speed": speed_flat,
            "accel": accel_flat,
            "p85_speed": p85_arr,
            "z_scalar": z_scalar,
            "z_sigma": z_scalar / sigma,
            "z_vector": z_vec.tolist(),
            "num_scenarios": len(unique_files),
        }

    # ── Run evaluations ──────────────────────────────────────────────────────
    results = {}
    t0 = time.time()
    for label, z_scalar in z_scalars.items():
        print(f"\n{'='*60}")
        print(f"Running {label}: z_scalar={z_scalar:.4f} ({z_scalar/sigma:.1f}σ)")
        print(f"{'='*60}")
        t_z = time.time()
        data = run_for_z(label, z_scalar)
        data["elapsed_s"] = time.time() - t_z
        results[label] = data
        print(f"  Elapsed: {data['elapsed_s']:.1f}s")

    vecenv.close()
    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s")

    # ── Save raw data ────────────────────────────────────────────────────────
    for label, data in results.items():
        np.savez_compressed(
            os.path.join(output_dir, f"{label}_raw.npz"),
            speed=data["speed"],
            accel=data["accel"],
            p85_speed=data["p85_speed"],
        )

    # Save metadata
    meta = {
        "pc1": pc1.tolist(),
        "sigma": sigma,
        "z_scalars": {k: v for k, v in z_scalars.items()},
        "checkpoint": CHECKPOINT,
        "target_scenarios": target_scenarios,
        "total_time_s": total_time,
        "timestamp": timestamp,
    }
    for label, data in results.items():
        meta[label] = {
            "z_scalar": data["z_scalar"],
            "z_sigma": data["z_sigma"],
            "z_vector": data["z_vector"],
            "num_scenarios": data["num_scenarios"],
            "n_speed_samples": len(data["speed"]),
            "n_accel_samples": len(data["accel"]),
            "n_p85_samples": len(data["p85_speed"]),
            "mean_speed": float(np.mean(data["speed"])),
            "mean_accel": float(np.mean(data["accel"])),
            "mean_abs_accel": float(np.mean(np.abs(data["accel"]))),
            "mean_p85_speed": float(np.mean(data["p85_speed"])),
            "elapsed_s": data["elapsed_s"],
        }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nData saved to: {output_dir}")
    for label in results:
        print(f"  {label}_raw.npz: speed={len(results[label]['speed'])}, "
              f"accel={len(results[label]['accel'])}, "
              f"p85={len(results[label]['p85_speed'])}")

    # ── Quick summary stats ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for label, data in results.items():
        s, a, p = data["speed"], data["accel"], data["p85_speed"]
        print(f"{label} (z={data['z_sigma']:.1f}σ):")
        print(f"  Speed:  mean={np.mean(s):.2f}, std={np.std(s):.2f}, "
              f"median={np.median(s):.2f}, p85={np.percentile(s, 85):.2f}")
        print(f"  Accel:  mean={np.mean(a):.2f}, std={np.std(a):.2f}, "
              f"mean|a|={np.mean(np.abs(a)):.2f}")
        print(f"  P85/agent: mean={np.mean(p):.2f}, std={np.std(p):.2f}")


if __name__ == "__main__":
    main()
