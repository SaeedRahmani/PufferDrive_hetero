"""Pre-training pipeline for 4D VAE style conditioning (vae_randomz_bc).

Steps:
  1. Initialize environment to collect expert data from binary maps
  2. Train VAE (z_dim=4, beta=0.1) on expert continuous actions
  3. Label expert sequences with trained VAE (expert_style_z_h32.pt)
  4. Compute PCA on z labels and save pc1_vector.npy

This script must be run from the project root directory:
  cd /scratch/srahmani/pufferdrive/pufferdrive_vae_randomz_bc
  python pretrain_vae_pipeline.py

It requires:
  - Binary maps at resources/drive/binaries/training/
  - Working pufferlib installation with C bindings
"""

import os
import sys
import argparse
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Where expert data lives (matches drive.ini: human_data_dir)
EXPERT_DATA_DIR = os.path.join(PROJECT_ROOT, "resources", "drive", "human_demonstrations")

# Where VAE model is loaded from during training (hardcoded in drive.py)
VAE_MODEL_DIR = os.path.join(PROJECT_ROOT, "pufferlib", "resources", "drive", "human_demonstrations")


def step1_collect_expert_data(num_maps=9000, num_agents=1024, bptt_horizon=32,
                               max_expert_sequences=3600):
    """Collect expert data by running C simulator on binary maps."""
    print("=" * 60)
    print("STEP 1: Collecting expert data from binary maps")
    print(f"  num_maps={num_maps}, num_agents={num_agents}")
    print(f"  bptt_horizon={bptt_horizon}, max_expert_sequences={max_expert_sequences}")
    print(f"  output_dir={EXPERT_DATA_DIR}")
    print("=" * 60)

    from pufferlib.ocean.drive.drive import Drive

    # Reset class-level flag to force re-collection
    Drive._human_data_prepped = False

    os.makedirs(EXPERT_DATA_DIR, exist_ok=True)

    env = Drive(
        num_maps=num_maps,
        num_agents=num_agents,
        bptt_horizon=bptt_horizon,
        human_data_dir=EXPERT_DATA_DIR,
        max_expert_sequences=max_expert_sequences,
        prep_human_data=True,
        save_data_to_disk=True,
        control_mode="control_vehicles",
        map_dir="resources/drive/binaries/training",
        style_z_dim=4,
        style_z_dropout_prob=0.5,
        episode_length=91,
        init_steps=0,
        action_type="discrete",
        dynamics_model="classic",
    )

    metrics = env.expert_data_metrics
    print("\nExpert data collection metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")

    # Verify files exist
    for fname in [
        f"expert_actions_discrete_h{bptt_horizon}.pt",
        f"expert_actions_continuous_h{bptt_horizon}.pt",
        f"expert_observations_h{bptt_horizon}.pt",
    ]:
        path = os.path.join(EXPERT_DATA_DIR, fname)
        assert os.path.exists(path), f"Missing: {path}"
        t = torch.load(path, map_location="cpu", weights_only=False)
        print(f"  {fname}: shape={t.shape}, dtype={t.dtype}")

    env.close()
    print("\nStep 1 complete.\n")
    return metrics


def step2_train_vae(z_dim=4, beta=0.1, hidden_size=64, epochs=200,
                     lr=1e-3, batch_size=256, bptt_horizon=32, device="cuda"):
    """Train trajectory VAE on collected expert data."""
    print("=" * 60)
    print(f"STEP 2: Training VAE (z_dim={z_dim}, beta={beta}, epochs={epochs})")
    print(f"  data_dir={EXPERT_DATA_DIR}")
    print(f"  device={device}")
    print("=" * 60)

    from pufferlib.ocean.drive.trajectory_vae import train_vae

    # Train VAE — saves vae_model.pt and vae_feature_stats.pt to EXPERT_DATA_DIR
    model_path = train_vae(
        data_dir=EXPERT_DATA_DIR,
        z_dim=z_dim,
        beta=beta,
        hidden_size=hidden_size,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        bptt_horizon=bptt_horizon,
        device=device,
    )

    print(f"\nVAE model saved to: {model_path}")

    # Copy VAE model and stats to the location where drive.py loads them
    # (only if the two directories are not the same via symlinks)
    os.makedirs(VAE_MODEL_DIR, exist_ok=True)
    import shutil

    src_real = os.path.realpath(EXPERT_DATA_DIR)
    dst_real = os.path.realpath(VAE_MODEL_DIR)
    if src_real != dst_real:
        for fname in ["vae_model.pt", "vae_feature_stats.pt"]:
            src = os.path.join(EXPERT_DATA_DIR, fname)
            dst = os.path.join(VAE_MODEL_DIR, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"  Copied {fname} -> {dst}")
    else:
        print("  EXPERT_DATA_DIR and VAE_MODEL_DIR resolve to same path; no copy needed.")

    print("\nStep 2 complete.\n")
    return model_path


def step3_label_data(bptt_horizon=32, device="cuda"):
    """Label expert sequences with trained VAE to produce expert_style_z_h32.pt."""
    print("=" * 60)
    print("STEP 3: Labeling expert data with VAE z-vectors")
    print(f"  data_dir={EXPERT_DATA_DIR}")
    print("=" * 60)

    from pufferlib.ocean.drive.trajectory_vae import label_expert_data

    vae_model_path = os.path.join(EXPERT_DATA_DIR, "vae_model.pt")
    assert os.path.exists(vae_model_path), f"VAE model not found: {vae_model_path}"

    z_path = label_expert_data(
        vae_model_path=vae_model_path,
        data_dir=EXPERT_DATA_DIR,
        bptt_horizon=bptt_horizon,
        device=device,
    )

    print(f"\nLabeled z saved to: {z_path}")

    # Verify
    z_data = torch.load(z_path, map_location="cpu", weights_only=False)
    print(f"  Shape: {z_data.shape}")
    print(f"  Mean:  {z_data.mean(dim=0).numpy()}")
    print(f"  Std:   {z_data.std(dim=0).numpy()}")
    print(f"  Min:   {z_data.min(dim=0).values.numpy()}")
    print(f"  Max:   {z_data.max(dim=0).values.numpy()}")

    print("\nStep 3 complete.\n")
    return z_path


def step4_compute_pca(bptt_horizon=32):
    """Compute PCA on z labels and save pc1_vector.npy."""
    print("=" * 60)
    print("STEP 4: Computing PCA on expert z-vectors")
    print("=" * 60)

    z_path = os.path.join(EXPERT_DATA_DIR, f"expert_style_z_h{bptt_horizon}.pt")
    assert os.path.exists(z_path), f"Z data not found: {z_path}"

    z_data = torch.load(z_path, map_location="cpu", weights_only=False).numpy()
    print(f"  Z data shape: {z_data.shape}")

    # Center the data
    z_mean = z_data.mean(axis=0)
    z_centered = z_data - z_mean

    # Compute covariance and eigenvectors
    cov = np.cov(z_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # PC1 = first eigenvector (unit norm)
    pc1 = eigvecs[:, 0]
    pc1 = pc1 / np.linalg.norm(pc1)

    # Variance explained
    var_ratio = eigvals / eigvals.sum()

    print(f"\n  Eigenvalues: {eigvals}")
    print(f"  Variance explained: {var_ratio}")
    print(f"  PC1 direction: {pc1}")
    print(f"  PC1 explains {var_ratio[0]*100:.1f}% of variance")

    # Save PC1 vector
    pc1_path = os.path.join(PROJECT_ROOT, "pc1_vector.npy")
    np.save(pc1_path, pc1)
    print(f"\n  Saved PC1 to: {pc1_path}")

    # Also compute the sweep range (mean ± 3σ of PC1 projections)
    projections = z_centered @ pc1
    proj_mean = projections.mean()
    proj_std = projections.std()
    sweep_lo = proj_mean - 3 * proj_std
    sweep_hi = proj_mean + 3 * proj_std
    print(f"\n  PC1 projection stats:")
    print(f"    Mean: {proj_mean:.4f}")
    print(f"    Std:  {proj_std:.4f}")
    print(f"    Suggested sweep range: [{sweep_lo:.4f}, {sweep_hi:.4f}]")

    print("\nStep 4 complete.\n")
    return pc1_path


def main():
    parser = argparse.ArgumentParser(description="VAE pre-training pipeline for vae_randomz_bc")
    parser.add_argument("--step", type=int, default=0,
                        help="Run a specific step (1-4). 0 = run all steps.")
    parser.add_argument("--device", default="cuda",
                        help="Device for VAE training (cuda or cpu)")
    parser.add_argument("--z-dim", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num-maps", type=int, default=9000)
    parser.add_argument("--max-expert-sequences", type=int, default=3600)
    args = parser.parse_args()

    # Ensure we're running from project root
    os.chdir(PROJECT_ROOT)
    print(f"Working directory: {os.getcwd()}")
    print(f"Expert data dir:   {EXPERT_DATA_DIR}")
    print(f"VAE model dir:     {VAE_MODEL_DIR}")
    print()

    if args.step == 0 or args.step == 1:
        step1_collect_expert_data(
            num_maps=args.num_maps,
            max_expert_sequences=args.max_expert_sequences,
        )

    if args.step == 0 or args.step == 2:
        step2_train_vae(
            z_dim=args.z_dim,
            beta=args.beta,
            epochs=args.epochs,
            device=args.device,
        )

    if args.step == 0 or args.step == 3:
        step3_label_data(device=args.device)

    if args.step == 0 or args.step == 4:
        step4_compute_pca()

    print("=" * 60)
    print("ALL STEPS COMPLETE")
    print("=" * 60)
    print(f"\nFiles produced:")
    for d in [EXPERT_DATA_DIR, VAE_MODEL_DIR]:
        if os.path.exists(d):
            for f in sorted(os.listdir(d)):
                fpath = os.path.join(d, f)
                size = os.path.getsize(fpath)
                print(f"  {fpath} ({size:,} bytes)")
    pc1_path = os.path.join(PROJECT_ROOT, "pc1_vector.npy")
    if os.path.exists(pc1_path):
        print(f"  {pc1_path} ({os.path.getsize(pc1_path):,} bytes)")


if __name__ == "__main__":
    main()
