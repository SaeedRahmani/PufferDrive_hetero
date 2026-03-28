"""Precompute VAE style z scores for ALL maps (training + validation).

Processes each map individually via symlinks to extract per-agent z values,
then builds a quantile normalization from the full distribution and stores
a lookup table for efficient loading during training.

Output files (in human_data_dir):
  - precomputed_z_training.pt:  {map_idx: {agent_id: z_norm}} for 9k training maps
  - precomputed_z_validation.pt: {map_idx: {agent_id: z_norm}} for 1k validation maps
  - z_quantile_reference.pt: sorted raw z values for runtime normalization
  - precomputed_z_raw_all.pt: raw (un-normalized) z per (map_idx, agent_id) for debugging

Usage:
  python precompute_all_style_z.py [--device cuda]
"""

import os
import sys
import time
import tempfile
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_vae(vae_model_path, device="cpu"):
    """Load trained VAE model."""
    from pufferlib.ocean.drive.trajectory_vae import TrajectoryVAE

    checkpoint = torch.load(vae_model_path, map_location=device, weights_only=False)
    vae = TrajectoryVAE(
        input_dim=checkpoint.get("input_dim", 4),
        z_dim=checkpoint.get("z_dim", 1),
        hidden_size=checkpoint.get("hidden_size", 64),
        seq_len=checkpoint.get("seq_len", 91),
    )
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae = vae.to(device)
    vae.eval()
    print(f"Loaded VAE: z_dim={vae.z_dim}, device={device}")
    return vae


def process_single_map(map_file_path, temp_dir, vae, feat_mean, feat_std, device="cpu"):
    """Process a single map file and return {agent_id: raw_z_value} for vehicles.

    Args:
        map_file_path: absolute path to map_XXX.bin
        temp_dir: temporary directory for symlinking
        vae: loaded VAE model
        feat_mean, feat_std: feature normalization stats
        device: torch device

    Returns:
        dict: {agent_id (int): raw_z (float)} for vehicle agents
        int: total agents in this map
        int: number of vehicles
    """
    from pufferlib.ocean.drive.drive import Drive
    from pufferlib.ocean.drive.trajectory_vae import extract_kinematic_features
    from pufferlib.ocean.drive import binding

    # Symlink the map file as map_000.bin in temp dir
    symlink_path = os.path.join(temp_dir, "map_000.bin")
    if os.path.exists(symlink_path):
        os.unlink(symlink_path)
    os.symlink(map_file_path, symlink_path)

    # Init env with just this one map, enough agents for all vehicles
    Drive._human_data_prepped = False
    try:
        env = Drive(
            num_maps=1,
            num_agents=128,  # enough for any single scenario
            bptt_horizon=32,
            human_data_dir=os.path.join(temp_dir, "hdata"),
            max_expert_sequences=100,
            prep_human_data=False,
            save_data_to_disk=False,
            control_mode="control_vehicles",
            map_dir=temp_dir,
            style_z_dim=1,
            episode_length=91,
            init_steps=0,
        )
    except Exception as e:
        # Some maps may fail to init (corrupt or empty)
        return {}, 0, 0

    num_agents = env.num_agents

    # Get GT trajectories for agent IDs and vehicle mask
    gt = env.get_ground_truth_trajectories()
    agent_ids = gt["id"].flatten()
    is_vehicle = gt["is_vehicle"].flatten()

    # Collect expert actions
    trajectory_length = 91
    expert_actions_continuous = np.zeros((trajectory_length, num_agents, 2), dtype=np.float32)
    expert_actions_discrete = np.zeros((trajectory_length, num_agents, 1), dtype=np.float32)
    expert_observations = np.zeros((trajectory_length, num_agents, env.num_obs), dtype=np.float32)

    binding.vec_collect_expert_data(
        env.c_envs, expert_actions_discrete, expert_actions_continuous, expert_observations
    )

    env.close()

    # Find unique vehicle agents (avoid duplicates from replicated envs)
    seen_agent_ids = set()
    unique_vehicle_indices = []
    for idx in range(num_agents):
        aid = int(agent_ids[idx])
        if is_vehicle[idx] and aid not in seen_agent_ids:
            # Check if this agent has valid data (not all -1)
            if not np.all(expert_actions_discrete[:, idx, 0] == -1.0):
                seen_agent_ids.add(aid)
                unique_vehicle_indices.append(idx)

    if not unique_vehicle_indices:
        return {}, num_agents, 0

    # Extract kinematic features for unique vehicles
    actions_tensor = torch.tensor(
        expert_actions_continuous[:, unique_vehicle_indices, :], dtype=torch.float32
    ).permute(1, 0, 2)  # (n_vehicles, traj_len, 2)

    features = extract_kinematic_features(actions_tensor)
    features = (features - feat_mean) / (feat_std + 1e-8)

    # Encode with VAE
    with torch.no_grad():
        z_raw = vae.encode(features.to(device), deterministic=True).cpu().numpy().flatten()

    # Build result: {agent_id: raw_z}
    result = {}
    for i, idx in enumerate(unique_vehicle_indices):
        aid = int(agent_ids[idx])
        result[aid] = float(z_raw[i])

    return result, num_agents, len(unique_vehicle_indices)


def main():
    parser = argparse.ArgumentParser(description="Precompute style z for all maps")
    parser.add_argument("--device", default="cuda", help="torch device")
    parser.add_argument("--training-dir", default="resources/drive/binaries/training")
    parser.add_argument("--validation-dir", default="resources/drive/binaries/validation")
    parser.add_argument("--human-data-dir", default="pufferlib/resources/drive/human_demonstrations")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load VAE model
    vae_model_path = os.path.join(args.human_data_dir, "vae_model.pt")
    if not os.path.exists(vae_model_path):
        print(f"ERROR: VAE model not found at {vae_model_path}")
        sys.exit(1)
    vae = load_vae(vae_model_path, device)

    # Load feature normalization stats
    stats_path = os.path.join(args.human_data_dir, "vae_feature_stats.pt")
    if not os.path.exists(stats_path):
        print(f"ERROR: Feature stats not found at {stats_path}")
        sys.exit(1)
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    feat_mean = stats["mean"]
    feat_std = stats["std"]

    # Process both training and validation maps
    datasets = [
        ("training", args.training_dir),
        ("validation", args.validation_dir),
    ]

    all_raw_z_values = []  # For building global quantile reference
    all_results = {}  # {dataset_name: {map_idx: {agent_id: raw_z}}}

    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.join(temp_dir, "hdata"), exist_ok=True)

        for dataset_name, map_dir in datasets:
            map_dir = os.path.abspath(map_dir)
            map_files = sorted([f for f in os.listdir(map_dir) if f.endswith(".bin")])
            num_maps = len(map_files)
            print(f"\n{'='*60}")
            print(f"Processing {dataset_name}: {num_maps} maps from {map_dir}")
            print(f"{'='*60}")

            dataset_results = {}
            total_vehicles = 0
            total_agents = 0
            failed_maps = 0
            t_start = time.time()

            for map_idx, map_file in enumerate(map_files):
                map_path = os.path.join(map_dir, map_file)

                agent_z, n_agents, n_vehicles = process_single_map(
                    map_path, temp_dir, vae, feat_mean, feat_std, device
                )

                if agent_z:
                    dataset_results[map_idx] = agent_z
                    all_raw_z_values.extend(agent_z.values())
                    total_vehicles += n_vehicles
                else:
                    failed_maps += 1

                total_agents += n_agents

                # Progress every 500 maps
                if (map_idx + 1) % 500 == 0 or map_idx == num_maps - 1:
                    elapsed = time.time() - t_start
                    rate = (map_idx + 1) / elapsed
                    eta = (num_maps - map_idx - 1) / rate if rate > 0 else 0
                    print(f"  [{map_idx+1:5d}/{num_maps}] "
                          f"{total_vehicles} vehicles, {failed_maps} failed | "
                          f"{rate:.1f} maps/s, ETA {eta:.0f}s")

            all_results[dataset_name] = dataset_results
            elapsed = time.time() - t_start
            print(f"\n{dataset_name} complete: {num_maps} maps, {total_vehicles} unique vehicles, "
                  f"{failed_maps} failed, {elapsed:.1f}s")

    # Build global quantile normalization reference
    print(f"\n{'='*60}")
    print(f"Building quantile normalization from {len(all_raw_z_values)} raw z values")
    print(f"{'='*60}")

    all_raw_z = np.array(all_raw_z_values, dtype=np.float32)
    z_sorted = np.sort(all_raw_z)
    print(f"Raw z: mean={all_raw_z.mean():.4f}, std={all_raw_z.std():.4f}, "
          f"range=[{all_raw_z.min():.4f}, {all_raw_z.max():.4f}]")

    # Apply quantile normalization to all results
    def quantile_normalize(raw_z_val):
        rank = np.searchsorted(z_sorted, raw_z_val, side="right")
        pct = rank / len(z_sorted)
        return float(pct * 2.0 - 1.0)

    normalized_results = {}
    for dataset_name, dataset_data in all_results.items():
        norm_data = {}
        for map_idx, agent_dict in dataset_data.items():
            norm_data[map_idx] = {
                aid: quantile_normalize(raw_z) for aid, raw_z in agent_dict.items()
            }
        normalized_results[dataset_name] = norm_data

    # Verify normalization
    all_norm_z = []
    for dataset_data in normalized_results.values():
        for agent_dict in dataset_data.values():
            all_norm_z.extend(agent_dict.values())
    all_norm_z = np.array(all_norm_z)
    print(f"Normalized z: mean={all_norm_z.mean():.4f}, std={all_norm_z.std():.4f}, "
          f"range=[{all_norm_z.min():.4f}, {all_norm_z.max():.4f}]")

    # Check uniformity
    n_bins = 10
    hist, _ = np.histogram(all_norm_z, bins=np.linspace(-1, 1, n_bins + 1))
    print(f"Distribution in {n_bins} bins:")
    for i in range(n_bins):
        bar = "#" * int(hist[i] / max(hist) * 30)
        print(f"  [{-1+i*0.2:+.1f}, {-1+(i+1)*0.2:+.1f}): {hist[i]:5d} {bar}")

    # Save everything
    os.makedirs(args.human_data_dir, exist_ok=True)

    # Normalized lookup tables
    torch.save(
        normalized_results["training"],
        os.path.join(args.human_data_dir, "precomputed_z_training.pt"),
    )
    print(f"Saved training lookup: {len(normalized_results['training'])} maps")

    torch.save(
        normalized_results["validation"],
        os.path.join(args.human_data_dir, "precomputed_z_validation.pt"),
    )
    print(f"Saved validation lookup: {len(normalized_results['validation'])} maps")

    # Sorted raw z for runtime quantile normalization of any new values
    torch.save(
        torch.from_numpy(z_sorted),
        os.path.join(args.human_data_dir, "z_quantile_reference.pt"),
    )
    print(f"Saved quantile reference: {len(z_sorted)} sorted z values")

    # Raw results for debugging
    raw_all = {}
    for dataset_name, dataset_data in all_results.items():
        for map_idx, agent_dict in dataset_data.items():
            key = f"{dataset_name}_{map_idx}"
            raw_all[key] = agent_dict
    torch.save(raw_all, os.path.join(args.human_data_dir, "precomputed_z_raw_all.pt"))
    print(f"Saved raw z debug file")

    print(f"\nDone! All files saved to {args.human_data_dir}")


if __name__ == "__main__":
    main()
