"""
Plot distribution comparisons: acceleration and P85 speed
for z=-2σ vs z=+2σ along PC1.

Usage:
    python plot_distribution_comparison.py --data-dir evaluation_results/dist_comparison_XXXXXX
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
})

# Colors
C_NEG = "#2166ac"  # blue = cautious / negative z
C_POS = "#b2182b"  # red = aggressive / positive z


def smooth_hist(data, bins, sigma_bins=2):
    """Compute a smoothed density curve from histogram counts using Gaussian convolution."""
    counts, edges = np.histogram(data, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kernel_size = int(6 * sigma_bins + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    x_k = np.arange(kernel_size) - kernel_size // 2
    kernel = np.exp(-0.5 * (x_k / sigma_bins) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(counts, kernel, mode="same")
    return centers, smoothed


def load_data(data_dir):
    neg = np.load(os.path.join(data_dir, "z_neg2_raw.npz"))
    pos = np.load(os.path.join(data_dir, "z_pos2_raw.npz"))
    with open(os.path.join(data_dir, "metadata.json")) as f:
        meta = json.load(f)
    return neg, pos, meta


def plot_accel_distribution(neg, pos, meta, fig_dir):
    """Smoothed density curves of acceleration distributions (no histogram bars)."""
    a_neg = neg["accel"]
    a_pos = pos["accel"]

    clip = 8.0
    a_neg_c = np.clip(a_neg, -clip, clip)
    a_pos_c = np.clip(a_pos, -clip, clip)

    z_neg_sigma = meta["z_neg2"]["z_sigma"]
    z_pos_sigma = meta["z_pos2"]["z_sigma"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Use heavy smoothing (sigma_bins=6) to hide discrete action-space artifacts
    bins = np.linspace(-clip, clip, 200)
    cx_neg, sy_neg = smooth_hist(a_neg_c, bins, sigma_bins=6)
    cx_pos, sy_pos = smooth_hist(a_pos_c, bins, sigma_bins=6)

    ax.fill_between(cx_neg, sy_neg, alpha=0.3, color=C_NEG)
    ax.plot(cx_neg, sy_neg, color=C_NEG, linewidth=2.5, alpha=0.9,
            label=f"$z = {z_neg_sigma:.1f}\\sigma$ (cautious)")
    ax.fill_between(cx_pos, sy_pos, alpha=0.3, color=C_POS)
    ax.plot(cx_pos, sy_pos, color=C_POS, linewidth=2.5, alpha=0.9,
            label=f"$z = +{z_pos_sigma:.1f}\\sigma$ (aggressive)")

    ax.axvline(np.mean(a_neg), color=C_NEG, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(np.mean(a_pos), color=C_POS, linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Acceleration (m/s²)")
    ax.set_ylabel("Density")
    ax.set_title("Acceleration Distribution: Cautious vs. Aggressive Style")
    ax.legend(loc="upper right")
    ax.set_xlim(-clip, clip)

    stats = (
        f"Cautious: μ={np.mean(a_neg):.2f}, σ={np.std(a_neg):.2f}, "
        f"|a|={np.mean(np.abs(a_neg)):.2f}\n"
        f"Aggressive: μ={np.mean(a_pos):.2f}, σ={np.std(a_pos):.2f}, "
        f"|a|={np.mean(np.abs(a_pos)):.2f}"
    )
    ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4",
            facecolor="wheat", alpha=0.6))

    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(fig_dir, f"fig_dist_acceleration.{fmt}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_dist_acceleration.png/.pdf")


def plot_p85_distribution(neg, pos, meta, fig_dir):
    """Overlapping histograms + smoothed density of per-agent P85 speed."""
    p_neg = neg["p85_speed"]
    p_pos = pos["p85_speed"]

    z_neg_sigma = meta["z_neg2"]["z_sigma"]
    z_pos_sigma = meta["z_pos2"]["z_sigma"]

    fig, ax = plt.subplots(figsize=(8, 5))

    lo = min(np.percentile(p_neg, 1), np.percentile(p_pos, 1))
    hi = max(np.percentile(p_neg, 99), np.percentile(p_pos, 99))
    bins = np.linspace(lo, hi, 100)

    ax.hist(p_neg, bins=bins, density=True, alpha=0.4, color=C_NEG,
            label=f"$z = {z_neg_sigma:.1f}\\sigma$ (cautious)", edgecolor="none")
    ax.hist(p_pos, bins=bins, density=True, alpha=0.4, color=C_POS,
            label=f"$z = +{z_pos_sigma:.1f}\\sigma$ (aggressive)", edgecolor="none")

    cx_neg, sy_neg = smooth_hist(p_neg, bins, sigma_bins=3)
    cx_pos, sy_pos = smooth_hist(p_pos, bins, sigma_bins=3)
    ax.plot(cx_neg, sy_neg, color=C_NEG, linewidth=2.5, alpha=0.9)
    ax.plot(cx_pos, sy_pos, color=C_POS, linewidth=2.5, alpha=0.9)

    ax.axvline(np.mean(p_neg), color=C_NEG, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(np.mean(p_pos), color=C_POS, linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("P85 Speed (m/s)")
    ax.set_ylabel("Density")
    ax.set_title("Per-Agent P85 Speed Distribution: Cautious vs. Aggressive Style")
    ax.legend(loc="upper right")

    stats = (
        f"Cautious: μ={np.mean(p_neg):.2f}, σ={np.std(p_neg):.2f}, "
        f"n={len(p_neg)}\n"
        f"Aggressive: μ={np.mean(p_pos):.2f}, σ={np.std(p_pos):.2f}, "
        f"n={len(p_pos)}"
    )
    ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4",
            facecolor="wheat", alpha=0.6))

    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(fig_dir, f"fig_dist_p85_speed.{fmt}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_dist_p85_speed.png/.pdf")


def plot_speed_distribution(neg, pos, meta, fig_dir):
    """Overlapping histograms + smoothed density of per-timestep speed."""
    s_neg = neg["speed"]
    s_pos = pos["speed"]

    z_neg_sigma = meta["z_neg2"]["z_sigma"]
    z_pos_sigma = meta["z_pos2"]["z_sigma"]

    fig, ax = plt.subplots(figsize=(8, 5))

    hi = max(np.percentile(s_neg, 99), np.percentile(s_pos, 99))
    bins = np.linspace(0, hi, 120)

    ax.hist(s_neg, bins=bins, density=True, alpha=0.4, color=C_NEG,
            label=f"$z = {z_neg_sigma:.1f}\\sigma$ (cautious)", edgecolor="none")
    ax.hist(s_pos, bins=bins, density=True, alpha=0.4, color=C_POS,
            label=f"$z = +{z_pos_sigma:.1f}\\sigma$ (aggressive)", edgecolor="none")

    cx_neg, sy_neg = smooth_hist(s_neg, bins, sigma_bins=3)
    cx_pos, sy_pos = smooth_hist(s_pos, bins, sigma_bins=3)
    ax.plot(cx_neg, sy_neg, color=C_NEG, linewidth=2.5, alpha=0.9)
    ax.plot(cx_pos, sy_pos, color=C_POS, linewidth=2.5, alpha=0.9)

    ax.axvline(np.mean(s_neg), color=C_NEG, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(np.mean(s_pos), color=C_POS, linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Density")
    ax.set_title("Speed Distribution: Cautious vs. Aggressive Style")
    ax.legend(loc="upper right")
    ax.set_xlim(0, hi)

    stats = (
        f"Cautious: μ={np.mean(s_neg):.2f}, p85={np.percentile(s_neg, 85):.2f}\n"
        f"Aggressive: μ={np.mean(s_pos):.2f}, p85={np.percentile(s_pos, 85):.2f}"
    )
    ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4",
            facecolor="wheat", alpha=0.6))

    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(fig_dir, f"fig_dist_speed.{fmt}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_dist_speed.png/.pdf")


def plot_combined_panel(neg, pos, meta, fig_dir):
    """2-panel figure: acceleration (left) and P85 speed (right)."""
    z_neg_sigma = meta["z_neg2"]["z_sigma"]
    z_pos_sigma = meta["z_pos2"]["z_sigma"]
    lbl_neg = f"$z = {z_neg_sigma:.1f}\\sigma$ (cautious)"
    lbl_pos = f"$z = +{z_pos_sigma:.1f}\\sigma$ (aggressive)"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: Acceleration (smooth curves only) ─────────────────────────
    a_neg = np.clip(neg["accel"], -8, 8)
    a_pos = np.clip(pos["accel"], -8, 8)
    bins_a = np.linspace(-8, 8, 200)
    cx_neg, sy_neg = smooth_hist(a_neg, bins_a, sigma_bins=6)
    cx_pos, sy_pos = smooth_hist(a_pos, bins_a, sigma_bins=6)
    ax1.fill_between(cx_neg, sy_neg, alpha=0.3, color=C_NEG)
    ax1.plot(cx_neg, sy_neg, color=C_NEG, lw=2.5, label=lbl_neg)
    ax1.fill_between(cx_pos, sy_pos, alpha=0.3, color=C_POS)
    ax1.plot(cx_pos, sy_pos, color=C_POS, lw=2.5, label=lbl_pos)
    ax1.axvline(np.mean(neg["accel"]), color=C_NEG, ls="--", lw=1.5, alpha=0.7)
    ax1.axvline(np.mean(pos["accel"]), color=C_POS, ls="--", lw=1.5, alpha=0.7)
    ax1.set_xlabel("Acceleration (m/s²)")
    ax1.set_ylabel("Density")
    ax1.set_title("(a) Acceleration Distribution")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.set_xlim(-8, 8)

    # ── Right: P85 Speed ─────────────────────────────────────────────────
    p_neg, p_pos = neg["p85_speed"], pos["p85_speed"]
    lo = min(np.percentile(p_neg, 1), np.percentile(p_pos, 1))
    hi = max(np.percentile(p_neg, 99), np.percentile(p_pos, 99))
    bins_p = np.linspace(lo, hi, 100)
    ax2.hist(p_neg, bins=bins_p, density=True, alpha=0.35, color=C_NEG,
             label=lbl_neg, edgecolor="none")
    ax2.hist(p_pos, bins=bins_p, density=True, alpha=0.35, color=C_POS,
             label=lbl_pos, edgecolor="none")
    cx_neg, sy_neg = smooth_hist(p_neg, bins_p, sigma_bins=3)
    cx_pos, sy_pos = smooth_hist(p_pos, bins_p, sigma_bins=3)
    ax2.plot(cx_neg, sy_neg, color=C_NEG, lw=2.5)
    ax2.plot(cx_pos, sy_pos, color=C_POS, lw=2.5)
    ax2.axvline(np.mean(p_neg), color=C_NEG, ls="--", lw=1.5, alpha=0.7)
    ax2.axvline(np.mean(p_pos), color=C_POS, ls="--", lw=1.5, alpha=0.7)
    ax2.set_xlabel("P85 Speed (m/s)")
    ax2.set_ylabel("Density")
    ax2.set_title("(b) Per-Agent P85 Speed Distribution")
    ax2.legend(loc="upper right", fontsize=10)

    fig.suptitle(
        "Behavioral Distribution Shift: Style-Conditioned Driving "
        "($\\pm 2\\sigma$ PC1)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(fig_dir, f"fig_dist_combined.{fmt}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_dist_combined.png/.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Dir with z_neg2_raw.npz, z_pos2_raw.npz, metadata.json")
    args = parser.parse_args()

    fig_dir = os.path.join(args.data_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Loading data from: {args.data_dir}")
    neg, pos, meta = load_data(args.data_dir)
    print(f"  z_neg2: {len(neg['speed'])} speed, {len(neg['accel'])} accel, "
          f"{len(neg['p85_speed'])} p85 samples")
    print(f"  z_pos2: {len(pos['speed'])} speed, {len(pos['accel'])} accel, "
          f"{len(pos['p85_speed'])} p85 samples")

    print("\nGenerating figures...")
    plot_accel_distribution(neg, pos, meta, fig_dir)
    plot_p85_distribution(neg, pos, meta, fig_dir)
    plot_speed_distribution(neg, pos, meta, fig_dir)
    plot_combined_panel(neg, pos, meta, fig_dir)

    print(f"\nAll figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
