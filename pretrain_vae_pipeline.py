"""Pre-training pipeline for 1D VAE style conditioning.

Steps:
  1. Initialize environment to collect vehicle-only expert data
  2. Train VAE (z_dim=1, beta=0.01) on expert continuous actions
  3. Label expert sequences with trained VAE
  4. Analyze VAE quality and style score distribution
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def step1_collect_expert_data(data_dir, num_maps=9000, num_agents=1024, bptt_horizon=32,
                               max_expert_sequences=3600):
    """Collect vehicle-only expert data by initializing the Drive environment."""
    print("=" * 60)
    print("STEP 1: Collecting vehicle-only expert data")
    print("=" * 60)

    from pufferlib.ocean.drive.drive import Drive

    # Reset class-level flag to force re-collection
    Drive._human_data_prepped = False

    env = Drive(
        num_maps=num_maps,
        num_agents=num_agents,
        bptt_horizon=bptt_horizon,
        human_data_dir=data_dir,
        max_expert_sequences=max_expert_sequences,
        prep_human_data=True,
        save_data_to_disk=True,
        control_mode="control_vehicles",
        map_dir="resources/drive/binaries/training",
        style_z_dim=4,  # 4D scene-conditioned VAE
        episode_length=91,
        init_steps=0,
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
        path = os.path.join(data_dir, fname)
        assert os.path.exists(path), f"Missing: {path}"
        t = torch.load(path, map_location="cpu", weights_only=False)
        print(f"  {fname}: shape={t.shape}, dtype={t.dtype}")

    env.close()
    print("\nStep 1 complete.\n")
    return metrics


def step2_train_vae(data_dir, z_dim=4, beta=0.1, epochs=200, bptt_horizon=32, device="cuda"):
    """Train scene-conditioned trajectory VAE on collected expert data."""
    print("=" * 60)
    print(f"STEP 2: Training scene-conditioned VAE (z_dim={z_dim}, beta={beta}, epochs={epochs})")
    print("=" * 60)

    from pufferlib.ocean.drive.trajectory_vae import train_vae

    model_path = train_vae(
        data_dir=data_dir,
        z_dim=z_dim,
        beta=beta,
        hidden_size=64,
        epochs=epochs,
        lr=1e-3,
        batch_size=256,
        bptt_horizon=bptt_horizon,
        device=device,
        ego_features=194,
        scene_dim=14,
    )

    print(f"\nVAE model saved to: {model_path}")
    print("Step 2 complete.\n")
    return model_path


def step3_label_data(vae_model_path, data_dir, bptt_horizon=32, device="cuda"):
    """Label expert sequences with trained VAE z vectors."""
    print("=" * 60)
    print("STEP 3: Labeling expert data with VAE z vectors")
    print("=" * 60)

    from pufferlib.ocean.drive.trajectory_vae import label_expert_data

    z_vectors = label_expert_data(
        vae_model_path=vae_model_path,
        data_dir=data_dir,
        bptt_horizon=bptt_horizon,
        device=device,
        ego_features=194,
    )

    print(f"\nLabeled {len(z_vectors)} sequences with z_dim={z_vectors.shape[1]}")
    print("Step 3 complete.\n")
    return z_vectors


def step4_analyze(data_dir, z_vectors, bptt_horizon=32, output_dir=None):
    """Analyze VAE quality and style score distribution."""
    print("=" * 60)
    print("STEP 4: Analyzing VAE quality and style scores")
    print("=" * 60)

    if output_dir is None:
        output_dir = os.path.join(data_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    z = z_vectors.numpy()  # (n_sequences, 1)
    z_flat = z.squeeze()   # (n_sequences,)

    # ---- Basic statistics ----
    print(f"\n--- Style Z Statistics (z_dim=1) ---")
    print(f"  N sequences:  {len(z_flat)}")
    print(f"  Mean:          {z_flat.mean():.4f}")
    print(f"  Std:           {z_flat.std():.4f}")
    print(f"  Min:           {z_flat.min():.4f}")
    print(f"  Max:           {z_flat.max():.4f}")
    print(f"  Median:        {np.median(z_flat):.4f}")
    print(f"  Skewness:      {scipy_stats.skew(z_flat):.4f}")
    print(f"  Kurtosis:      {scipy_stats.kurtosis(z_flat):.4f}")

    q25, q75 = np.percentile(z_flat, [25, 75])
    print(f"  Q25:           {q25:.4f}")
    print(f"  Q75:           {q75:.4f}")
    print(f"  IQR:           {q75 - q25:.4f}")

    # ---- Test: is z informative (not collapsed)? ----
    print(f"\n--- Posterior Collapse Check ---")
    # If VAE has collapsed, z will be ~N(0,1) with no correlation to data
    # A useful z should have std > 0.3 and not be exactly standard normal
    if z_flat.std() < 0.3:
        print("  WARNING: z std is very low — possible posterior collapse!")
        print("  The KL term may be too strong. Consider lowering beta further.")
    else:
        print(f"  z std = {z_flat.std():.4f} — looks healthy (not collapsed)")

    # Normality test — if z is perfectly N(0,1), the VAE isn't encoding anything
    ks_stat, ks_p = scipy_stats.kstest(z_flat, "norm")
    print(f"  KS test vs N(0,1): stat={ks_stat:.4f}, p={ks_p:.4f}")
    if ks_p > 0.05:
        print("  WARNING: z distribution is indistinguishable from N(0,1)")
        print("  This may indicate posterior collapse.")
    else:
        print("  z distribution differs from N(0,1) — VAE is encoding meaningful information")

    # ---- Correlation with kinematic features ----
    print(f"\n--- Correlation with Driving Style ---")
    from pufferlib.ocean.drive.trajectory_vae import extract_kinematic_features

    actions_path = os.path.join(data_dir, f"expert_actions_continuous_h{bptt_horizon}.pt")
    expert_actions = torch.load(actions_path, map_location="cpu", weights_only=False)
    features = extract_kinematic_features(expert_actions)  # (n, seq_len, 4)

    # Compute per-sequence summary stats
    feat_names = ["acceleration", "abs_steering", "approx_speed", "jerk"]
    feat_means = features.mean(dim=1).numpy()  # (n, 4)
    feat_stds = features.std(dim=1).numpy()     # (n, 4)

    print(f"\n  Pearson correlations between z and per-sequence feature means:")
    correlations = {}
    for i, name in enumerate(feat_names):
        r, p = scipy_stats.pearsonr(z_flat, feat_means[:, i])
        correlations[name + "_mean"] = (r, p)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    z vs mean_{name:15s}: r={r:+.4f}  p={p:.2e} {sig}")

    print(f"\n  Pearson correlations between z and per-sequence feature stds:")
    for i, name in enumerate(feat_names):
        r, p = scipy_stats.pearsonr(z_flat, feat_stds[:, i])
        correlations[name + "_std"] = (r, p)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    z vs std_{name:15s}:  r={r:+.4f}  p={p:.2e} {sig}")

    # ---- Z-binned behavior analysis ----
    print(f"\n--- Style Bins Analysis ---")
    n_bins = 5
    z_percentiles = np.percentile(z_flat, np.linspace(0, 100, n_bins + 1))
    bin_labels = []
    for b in range(n_bins):
        lo, hi = z_percentiles[b], z_percentiles[b + 1]
        mask = (z_flat >= lo) & (z_flat <= hi) if b == n_bins - 1 else (z_flat >= lo) & (z_flat < hi)
        n_in_bin = mask.sum()
        bin_labels.append(f"[{lo:.2f}, {hi:.2f}]")
        if n_in_bin == 0:
            continue
        print(f"\n  Bin {b+1} (z in {bin_labels[-1]}, n={n_in_bin}):")
        for i, name in enumerate(feat_names):
            m = feat_means[mask, i]
            print(f"    mean_{name:15s}: {m.mean():.4f} ± {m.std():.4f}")

    # ---- Plots ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("1D VAE Style Score Analysis", fontsize=14)

    # 1. Z histogram
    ax = axes[0, 0]
    ax.hist(z_flat, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    x_range = np.linspace(z_flat.min() - 0.5, z_flat.max() + 0.5, 200)
    ax.plot(x_range, scipy_stats.norm.pdf(x_range, 0, 1), "r--", label="N(0,1)")
    ax.set_xlabel("z value")
    ax.set_ylabel("Density")
    ax.set_title("Style z Distribution")
    ax.legend()

    # 2-5. Z vs each kinematic feature
    for i, name in enumerate(feat_names):
        row, col = divmod(i + 1, 3)
        ax = axes[row, col]
        # Subsample for scatter plot readability
        n_plot = min(2000, len(z_flat))
        idx = np.random.choice(len(z_flat), n_plot, replace=False)
        ax.scatter(z_flat[idx], feat_means[idx, i], alpha=0.15, s=8, color="steelblue")
        # Trend line
        slope, intercept, r, p, se = scipy_stats.linregress(z_flat, feat_means[:, i])
        ax.plot(x_range, slope * x_range + intercept, "r-", linewidth=2,
                label=f"r={r:.3f}")
        ax.set_xlabel("z")
        ax.set_ylabel(f"mean {name}")
        ax.set_title(f"z vs {name}")
        ax.legend()

    # 6. Box plot of z by behavior quintile
    ax = axes[1, 2]
    # Use the feature with strongest correlation
    best_feat_name = max(correlations, key=lambda k: abs(correlations[k][0]))
    best_feat_idx = feat_names.index(best_feat_name.replace("_mean", "").replace("_std", ""))
    is_mean = "_mean" in best_feat_name
    feat_vals = feat_means[:, best_feat_idx] if is_mean else feat_stds[:, best_feat_idx]
    quintile_edges = np.percentile(feat_vals, [0, 20, 40, 60, 80, 100])
    box_data = []
    box_labels = []
    for q in range(5):
        lo, hi = quintile_edges[q], quintile_edges[q + 1]
        mask = (feat_vals >= lo) & (feat_vals <= hi) if q == 4 else (feat_vals >= lo) & (feat_vals < hi)
        box_data.append(z_flat[mask])
        box_labels.append(f"Q{q+1}")
    ax.boxplot(box_data, labels=box_labels)
    ax.set_xlabel(f"{best_feat_name} quintile")
    ax.set_ylabel("z")
    ax.set_title(f"z by {best_feat_name} quintile")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "vae_1d_analysis.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\n  Analysis plot saved to: {plot_path}")
    plt.close()

    # ---- Reconstruction quality check ----
    print(f"\n--- Reconstruction Quality (sample check) ---")
    vae_model_path = os.path.join(data_dir, "vae_model.pt")
    checkpoint = torch.load(vae_model_path, map_location="cpu", weights_only=False)

    from pufferlib.ocean.drive.trajectory_vae import TrajectoryVAE
    vae = TrajectoryVAE(
        input_dim=checkpoint["input_dim"],
        z_dim=checkpoint["z_dim"],
        hidden_size=checkpoint["hidden_size"],
        seq_len=checkpoint["seq_len"],
    )
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()

    stats = torch.load(os.path.join(data_dir, "vae_feature_stats.pt"), map_location="cpu", weights_only=False)
    features_norm = (features - stats["mean"]) / stats["std"].clamp(min=1e-6)

    # Compute reconstruction error on a sample
    n_eval = min(1024, len(features_norm))
    with torch.no_grad():
        batch = features_norm[:n_eval]
        recon, mu, log_var = vae(batch)
        mse = ((recon - batch) ** 2).mean().item()
        # Per-feature MSE
        for i, name in enumerate(feat_names):
            feat_mse = ((recon[:, :, i] - batch[:, :, i]) ** 2).mean().item()
            print(f"  Recon MSE ({name:15s}): {feat_mse:.6f}")
        print(f"  Overall Recon MSE:          {mse:.6f}")

        # KL divergence
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean().item()
        print(f"  KL divergence:              {kl:.4f}")

    print(f"\n--- Summary ---")
    has_issues = False
    if z_flat.std() < 0.3:
        print("  [ISSUE] Possible posterior collapse — z std too low")
        has_issues = True
    if ks_p > 0.05:
        print("  [ISSUE] z indistinguishable from prior N(0,1)")
        has_issues = True
    
    sig_correlations = [(k, r, p) for k, (r, p) in correlations.items() if p < 0.01]
    if len(sig_correlations) == 0:
        print("  [ISSUE] No significant correlations between z and kinematic features")
        has_issues = True
    else:
        print(f"  [OK] {len(sig_correlations)} significant correlations found:")
        for k, r, p in sorted(sig_correlations, key=lambda x: -abs(x[1])):
            print(f"       z vs {k}: r={r:+.4f}")

    if not has_issues:
        print("  [OK] VAE appears to be encoding meaningful driving style information")

    print("\nStep 4 complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-training pipeline for 1D VAE")
    parser.add_argument("--data-dir", default="resources/drive/human_demonstrations")
    parser.add_argument("--z-dim", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--bptt-horizon", type=int, default=32)
    parser.add_argument("--num-maps", type=int, default=9000)
    parser.add_argument("--num-agents", type=int, default=1024)
    parser.add_argument("--max-expert-sequences", type=int, default=3600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-collect", action="store_true", help="Skip expert data collection")
    parser.add_argument("--skip-train", action="store_true", help="Skip VAE training")
    parser.add_argument("--skip-label", action="store_true", help="Skip labeling")
    parser.add_argument("--analysis-only", action="store_true", help="Run only analysis on existing data")
    args = parser.parse_args()

    data_dir = args.data_dir
    vae_model_path = os.path.join(data_dir, "vae_model.pt")

    if args.analysis_only:
        z_path = os.path.join(data_dir, f"expert_style_z_h{args.bptt_horizon}.pt")
        z_vectors = torch.load(z_path, map_location="cpu", weights_only=False)
        step4_analyze(data_dir, z_vectors, bptt_horizon=args.bptt_horizon)
        sys.exit(0)

    # Step 1: Collect expert data
    if not args.skip_collect:
        step1_collect_expert_data(
            data_dir=data_dir,
            num_maps=args.num_maps,
            num_agents=args.num_agents,
            bptt_horizon=args.bptt_horizon,
            max_expert_sequences=args.max_expert_sequences,
        )
    else:
        print("Skipping Step 1 (expert data collection)")

    # Step 2: Train VAE
    if not args.skip_train:
        vae_model_path = step2_train_vae(
            data_dir=data_dir,
            z_dim=args.z_dim,
            beta=args.beta,
            epochs=args.epochs,
            bptt_horizon=args.bptt_horizon,
            device=args.device,
        )
    else:
        print("Skipping Step 2 (VAE training)")

    # Step 3: Label expert data
    if not args.skip_label:
        z_vectors = step3_label_data(
            vae_model_path=vae_model_path,
            data_dir=data_dir,
            bptt_horizon=args.bptt_horizon,
            device=args.device,
        )
    else:
        print("Skipping Step 3 (labeling)")
        z_path = os.path.join(data_dir, f"expert_style_z_h{args.bptt_horizon}.pt")
        z_vectors = torch.load(z_path, map_location="cpu", weights_only=False)

    # Step 4: Analyze
    step4_analyze(data_dir, z_vectors, bptt_horizon=args.bptt_horizon)
