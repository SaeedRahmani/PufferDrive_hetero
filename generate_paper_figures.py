#!/usr/bin/env python3
"""Generate all paper figures from evaluation data."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

FIG_DIR = "paper_figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Colors ───────────────────────────────────────────────────────────────────
C_BLUE = "#2166ac"
C_RED = "#b2182b"
C_GREEN = "#1b7837"
C_ORANGE = "#e66101"
C_PURPLE = "#7b3294"
C_CYAN = "#01665e"
C_GOLD = "#d4a017"
C_GRAY = "#666666"

# =============================================================================
# Load 1000-scenario PC1 sweep data
# =============================================================================
SWEEP_JSON = "evaluation_results/full_eval_1000sc_20260413_021515/exp2_pc1/pufferdrive_vae_randomz_bc_sweep_ep005801_20260413_051642.json"
with open(SWEEP_JSON) as f:
    sweep_data = json.load(f)

z_labels = sorted(sweep_data["results"].keys(), key=lambda x: float(x.split("_")[1]))
z_vals = []
realism, ade, mean_speed, p85_speed, speed_var = [], [], [], [], []
mean_accel, accel_var, coll_rate, offroad_rate, completion = [], [], [], [], []
# WOSAC sub-scores
ll_speed, ll_accel, ll_angspeed, ll_angaccel = [], [], [], []
ll_dist_obj, ll_ttc, ll_coll_ind = [], [], []
ll_road_edge, ll_offroad_ind = [], []
kinematic_sub, interactive_sub, map_sub = [], [], []

for zk in z_labels:
    r = sweep_data["results"][zk]
    z_vals.append(r["z_value"])
    realism.append(r["realism_meta_score"])
    ade.append(r["ade"])
    mean_speed.append(r["behavioral_mean_speed"])
    p85_speed.append(r["behavioral_p85_speed"])
    speed_var.append(r["behavioral_speed_variance"])
    mean_accel.append(r["behavioral_mean_accel"])
    accel_var.append(r["behavioral_accel_variance"])
    coll_rate.append(r["behavioral_info_collision_rate"])
    offroad_rate.append(r["behavioral_info_offroad_rate"])
    completion.append(r["behavioral_info_completion_rate"])
    ll_speed.append(r["likelihood_linear_speed"])
    ll_accel.append(r["likelihood_linear_acceleration"])
    ll_angspeed.append(r["likelihood_angular_speed"])
    ll_angaccel.append(r["likelihood_angular_acceleration"])
    ll_dist_obj.append(r["likelihood_distance_to_nearest_object"])
    ll_ttc.append(r["likelihood_time_to_collision"])
    ll_coll_ind.append(r["likelihood_collision_indication"])
    ll_road_edge.append(r["likelihood_distance_to_road_edge"])
    ll_offroad_ind.append(r["likelihood_offroad_indication"])
    kinematic_sub.append(r["kinematic_metrics"])
    interactive_sub.append(r["interactive_metrics"])
    map_sub.append(r["map_based_metrics"])

z_vals = np.array(z_vals)
# Convert to sigma units
sigma_pc1 = 1.1428  # from PCA
z_sigma = z_vals / sigma_pc1

# x-tick labels
xtick_labels = [f"${v:.1f}\\sigma$" for v in z_sigma]


def save(fig, name):
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{fmt}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.png/.pdf")


# =============================================================================
# FIGURE 1: WOSAC + ADE side by side (realism metrics)
# =============================================================================
print("Figure 1: WOSAC + ADE")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# (a) WOSAC realism meta-score and sub-scores
ax1.plot(z_sigma, realism, "o-", color=C_BLUE, linewidth=2, markersize=6, label="Meta-score", zorder=5)
ax1.plot(z_sigma, kinematic_sub, "s--", color=C_ORANGE, linewidth=1.5, markersize=5, label="Kinematic", alpha=0.8)
ax1.plot(z_sigma, interactive_sub, "^--", color=C_GREEN, linewidth=1.5, markersize=5, label="Interactive", alpha=0.8)
ax1.plot(z_sigma, map_sub, "D--", color=C_PURPLE, linewidth=1.5, markersize=5, label="Map-based", alpha=0.8)
ax1.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
ax1.set_xlabel("Style $z$-value ($\\sigma$ along PC1)")
ax1.set_ylabel("WOSAC Likelihood Score")
ax1.set_title("(a) WOSAC Realism Scores")
ax1.legend(loc="lower left", fontsize=8)
ax1.set_xticks(z_sigma)
ax1.set_xticklabels(xtick_labels, rotation=45, fontsize=7)
ax1.set_ylim(0.15, 0.9)
# Add horizontal band for acceptable realism
ax1.axhspan(0.69, 0.72, alpha=0.08, color=C_BLUE, zorder=0)

# (b) ADE
ax2.plot(z_sigma, ade, "o-", color=C_RED, linewidth=2, markersize=6, zorder=5)
ax2.fill_between(z_sigma, ade, alpha=0.15, color=C_RED)
ax2.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
ax2.set_xlabel("Style $z$-value ($\\sigma$ along PC1)")
ax2.set_ylabel("Average Displacement Error (m)")
ax2.set_title("(b) ADE vs. Style")
ax2.set_xticks(z_sigma)
ax2.set_xticklabels(xtick_labels, rotation=45, fontsize=7)  

# Annotate minimum
min_idx = np.argmin(ade)
ax2.annotate(f"min={ade[min_idx]:.1f}m\n($z=0$)", xy=(z_sigma[min_idx], ade[min_idx]),
             xytext=(z_sigma[min_idx]+0.8, ade[min_idx]+0.8),
             arrowprops=dict(arrowstyle="->", color=C_GRAY), fontsize=8, color=C_GRAY)

fig.suptitle("Human Realism Metrics Across Style Sweep (1000 Scenarios)", fontsize=12, y=1.02)
plt.tight_layout()
save(fig, "fig_realism_ade")


# =============================================================================
# FIGURE 2: Behavioral metrics sweep
# =============================================================================
print("Figure 2: Behavioral metrics")
fig, axes = plt.subplots(2, 3, figsize=(12, 7))

# (a) Mean Speed
ax = axes[0, 0]
ax.plot(z_sigma, mean_speed, "o-", color=C_BLUE, linewidth=2, markersize=5)
ax.fill_between(z_sigma, mean_speed, alpha=0.15, color=C_BLUE)
ax.set_ylabel("Mean Speed (m/s)")
ax.set_title("(a) Mean Speed")
ax.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

# (b) Speed Variance
ax = axes[0, 1]
ax.plot(z_sigma, speed_var, "s-", color=C_GREEN, linewidth=2, markersize=5)
ax.fill_between(z_sigma, speed_var, alpha=0.15, color=C_GREEN)
ax.set_ylabel("Speed Variance (m$^2$/s$^2$)")
ax.set_title("(b) Speed Variance")
ax.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

# (c) Mean Acceleration
ax = axes[0, 2]
ax.plot(z_sigma, mean_accel, "^-", color=C_ORANGE, linewidth=2, markersize=5)
ax.fill_between(z_sigma, mean_accel, alpha=0.15, color=C_ORANGE)
ax.set_ylabel("Mean Acceleration (m/s$^2$)")
ax.set_title("(c) Mean Acceleration")
ax.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

# (d) Acceleration Variance
ax = axes[1, 0]
ax.plot(z_sigma, accel_var, "D-", color=C_PURPLE, linewidth=2, markersize=5)
ax.fill_between(z_sigma, accel_var, alpha=0.15, color=C_PURPLE)
ax.set_ylabel("Acceleration Variance")
ax.set_title("(d) Acceleration Variance")
ax.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

# (e) Collision Rate
ax = axes[1, 1]
ax.plot(z_sigma, [c*100 for c in coll_rate], "v-", color=C_RED, linewidth=2, markersize=5)
ax.fill_between(z_sigma, [c*100 for c in coll_rate], alpha=0.15, color=C_RED)
ax.set_ylabel("Collision Rate (%)")
ax.set_title("(e) Collision Rate")
ax.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

# (f) Off-road Rate
ax = axes[1, 2]
ax.plot(z_sigma, [o*100 for o in offroad_rate], "p-", color=C_CYAN, linewidth=2, markersize=5)
ax.fill_between(z_sigma, [o*100 for o in offroad_rate], alpha=0.15, color=C_CYAN)
ax.set_ylabel("Off-road Rate (%)")
ax.set_title("(f) Off-road Rate")
ax.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

for ax in axes.flat:
    ax.set_xlabel("Style $z$ ($\\sigma$ along PC1)")
    ax.set_xticks(z_sigma[::2])
    ax.set_xticklabels([xtick_labels[i] for i in range(0, len(xtick_labels), 2)], fontsize=7)

fig.suptitle("Behavioral Metrics Across PC1 Style Sweep (1000 Scenarios)", fontsize=13, y=1.01)
plt.tight_layout()
save(fig, "fig_behavioral_metrics")


# =============================================================================
# FIGURE 3: Ablation study - speed range across z sweep
# =============================================================================
print("Figure 3: Ablation study")

# Read the 9-project sweep CSV
import csv

ablation_sweep_file = "/scratch/srahmani/pufferdrive/evaluation_results/figures/table_exp2_sweep_88192819.csv"
ablation_baseline_file = "/scratch/srahmani/pufferdrive/evaluation_results/figures/table_exp1_baseline_88192819.csv"

# Parse sweep data per project
ablation_sweep = {}
with open(ablation_sweep_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        proj = row["project"]
        if proj not in ablation_sweep:
            ablation_sweep[proj] = {"z": [], "realism": [], "ade": [], "speed": []}
        ablation_sweep[proj]["z"].append(float(row["z_value"]))
        ablation_sweep[proj]["realism"].append(float(row["realism_meta_score"]))
        ablation_sweep[proj]["ade"].append(float(row["ade"]))
        ablation_sweep[proj]["speed"].append(float(row["mean_speed"]))

# Parse baseline data
ablation_baseline = {}
with open(ablation_baseline_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        ablation_baseline[row["project"]] = {
            "realism": float(row["realism_meta_score"]),
            "ade": float(row["ade"]),
        }

# Select key variants for the ablation figure
# Our model is VAE RandZ BC = "pufferdrive_vae_randomz_bc" but in the old sweep it's listed differently
# Let's use what we have: Hetero-v1, Hetero-v2, VAE (LL=1), VAE (LL=0), Causal (LL=1), Causal (LL=0)
ablation_projects = {
    "Hetero-v1": {"color": C_BLUE, "marker": "o", "label": "Hetero-v1 (disc.)"},
    "Hetero-v2": {"color": C_GREEN, "marker": "s", "label": "Hetero-v2 (disc.)"},
    "VAE (LL=0)": {"color": C_RED, "marker": "^", "label": "VAE (no BC, $\\lambda=0$)"},
    "VAE (LL=1)": {"color": C_ORANGE, "marker": "D", "label": "VAE (with BC, $\\lambda=1$)"},
    "Causal (LL=0)": {"color": C_PURPLE, "marker": "v", "label": "Causal (no BC)"},
    "Causal (LL=1)": {"color": C_CYAN, "marker": "p", "label": "Causal (with BC)"},
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

for proj, style in ablation_projects.items():
    if proj in ablation_sweep:
        d = ablation_sweep[proj]
        z_arr = np.array(d["z"])
        order = np.argsort(z_arr)
        ax1.plot(z_arr[order], np.array(d["realism"])[order], f'{style["marker"]}-',
                 color=style["color"], linewidth=1.5, markersize=5, label=style["label"], alpha=0.85)
        ax2.plot(z_arr[order], np.array(d["speed"])[order], f'{style["marker"]}-',
                 color=style["color"], linewidth=1.5, markersize=5, label=style["label"], alpha=0.85)

# Add our model (SCSP) from the 1000sc data, but rescaled to same z range [-1, 1]
# Filter to only z in [-1.71, 1.71] sigma (approx [-1, 1] in raw z)
mask = np.abs(z_vals) <= 2.0
ax1.plot(z_vals[mask], np.array(realism)[mask], "h-", color=C_GOLD, linewidth=2.5,
         markersize=7, label="SCSP (Ours)", zorder=10)
ax2.plot(z_vals[mask], np.array(mean_speed)[mask], "h-", color=C_GOLD, linewidth=2.5,
         markersize=7, label="SCSP (Ours)", zorder=10)

ax1.set_xlabel("Style $z$-value")
ax1.set_ylabel("WOSAC Realism Meta-Score")
ax1.set_title("(a) Realism Across Style Sweep")
ax1.legend(fontsize=7, loc="lower left", ncol=2)
ax1.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

ax2.set_xlabel("Style $z$-value")
ax2.set_ylabel("Mean Speed (m/s)")
ax2.set_title("(b) Speed Controllability")
ax2.legend(fontsize=7, loc="upper left", ncol=2)
ax2.axvline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

fig.suptitle("Ablation Study: Realism and Controllability Across Architectures", fontsize=12, y=1.02)
plt.tight_layout()
save(fig, "fig_ablation")


# =============================================================================
# FIGURE 4: Homogeneous vs Heterogeneous stress test
# =============================================================================
print("Figure 4: Hetero vs Homo")

EXP5_JSON = "evaluation_results/exp5/exp5_guided_vs_hetero_20260413_035338.json"
with open(EXP5_JSON) as f:
    exp5 = json.load(f)

homo = exp5["results"]["homogeneous"]
hetero = exp5["results"]["heterogeneous"]
logrep = exp5["results"]["log_replay"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# (a) Safety metrics (lower is better)
metrics_neg = ["Collision\nRate", "Off-road\nRate"]
homo_neg = [homo["behavioral_info_collision_rate"]*100, homo["behavioral_info_offroad_rate"]*100]
hetero_neg = [hetero["behavioral_info_collision_rate"]*100, hetero["behavioral_info_offroad_rate"]*100]
logrep_neg = [logrep["behavioral_info_collision_rate"]*100, logrep["behavioral_info_offroad_rate"]*100]

x = np.arange(len(metrics_neg))
w = 0.25
bars1 = ax1.bar(x - w, homo_neg, w, color=C_BLUE, alpha=0.8, label="Homogeneous ($z=0$)")
bars2 = ax1.bar(x, hetero_neg, w, color=C_RED, alpha=0.8, label="Heterogeneous ($z\\sim\\mathcal{N}$)")
bars3 = ax1.bar(x + w, logrep_neg, w, color=C_GRAY, alpha=0.6, label="Log Replay")

ax1.set_ylabel("Rate (%)")
ax1.set_title("(a) Safety Metrics (lower is better)")
ax1.set_xticks(x)
ax1.set_xticklabels(metrics_neg)
ax1.legend(fontsize=8)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}%", ha="center", va="bottom", fontsize=7)

# (b) Performance metrics (higher is better)
metrics_pos = ["Goal\nCompletion", "Overall\nScore"]
homo_pos = [homo["behavioral_info_completion_rate"]*100, homo["behavioral_info_score"]*100]
hetero_pos = [hetero["behavioral_info_completion_rate"]*100, hetero["behavioral_info_score"]*100]
logrep_pos = [logrep["behavioral_info_completion_rate"]*100, logrep["behavioral_info_score"]*100]

x = np.arange(len(metrics_pos))
bars1 = ax2.bar(x - w, homo_pos, w, color=C_BLUE, alpha=0.8, label="Homogeneous ($z=0$)")
bars2 = ax2.bar(x, hetero_pos, w, color=C_RED, alpha=0.8, label="Heterogeneous ($z\\sim\\mathcal{N}$)")
bars3 = ax2.bar(x + w, logrep_pos, w, color=C_GRAY, alpha=0.6, label="Log Replay")

ax2.set_ylabel("Rate (%)")
ax2.set_title("(b) Performance Metrics (higher is better)")
ax2.set_xticks(x)
ax2.set_xticklabels(metrics_pos)
ax2.legend(fontsize=8)
ax2.set_ylim(70, 100)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7)

fig.suptitle("Ego Policy Stress Test: Homogeneous vs. Heterogeneous Traffic", fontsize=12, y=1.02)
plt.tight_layout()
save(fig, "fig_hetero_homo")


# =============================================================================
# FIGURE 5: Sample scenarios showing z effect
# =============================================================================
print("Figure 5: Sample scenarios")

# We have trajectory snapshots for scenarios 000, 100, 200, etc.
# Each shows z=-0.95, z=0.00, z=+0.95
scenario_base = "evaluation_results/eval_200sc_20260412_163046/viz_z_sweep/uniform/vae_randomz_bc"
scenarios = ["scenario_val_map_000", "scenario_val_map_300", "scenario_val_map_700"]
mode = "sdc_only"

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for i, sc in enumerate(scenarios):
    img_path = os.path.join(scenario_base, sc, mode, "trajectory_snapshot.png")
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        axes[i].imshow(img)
        axes[i].set_title(f"Scenario {sc.split('_')[-1]}", fontsize=11)
    else:
        axes[i].text(0.5, 0.5, f"Image not found:\n{sc}", ha="center", va="center", transform=axes[i].transAxes)
    axes[i].axis("off")

fig.suptitle("Sample Scenarios: SDC Trajectory Under Different Style Conditions ($z=-0.95$, $z=0$, $z=+0.95$)",
             fontsize=11, y=1.0)
plt.tight_layout()
save(fig, "fig_sample_scenarios")


# =============================================================================
# FIGURE 6 (bonus): Framework diagram (simple text-based)
# =============================================================================
print("Figure 6: Framework diagram")

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

# Define boxes
boxes = {
    "Expert\nTrajectories\n(WOMD)": (0.5, 4.5, 1.8, 1.0),
    "VAE\nEncoder": (3.0, 4.5, 1.2, 1.0),
    "Style\nLatent $\\mathbf{z}$": (5.0, 4.5, 1.2, 1.0),
    "Shared RL Policy\n$\\pi_\\theta(a|o, \\mathbf{z})$": (5.0, 2.5, 2.0, 1.0),
    "GPUDrive\nSimulator": (8.0, 2.5, 1.5, 1.0),
    "PPO\n(RL Loss)": (2.0, 0.8, 1.5, 0.9),
    "BC Loss\n($\\mathcal{L}_{BC}$)": (5.0, 0.8, 1.5, 0.9),
    "Random $\\mathbf{z}$\n$\\sim \\mathcal{N}(0,I)$": (8.0, 4.5, 1.5, 1.0),
}

colors = {
    "Expert\nTrajectories\n(WOMD)": "#dae8fc",
    "VAE\nEncoder": "#d5e8d4",
    "Style\nLatent $\\mathbf{z}$": "#fff2cc",
    "Shared RL Policy\n$\\pi_\\theta(a|o, \\mathbf{z})$": "#f8cecc",
    "GPUDrive\nSimulator": "#e1d5e7",
    "PPO\n(RL Loss)": "#f8cecc",
    "BC Loss\n($\\mathcal{L}_{BC}$)": "#d5e8d4",
    "Random $\\mathbf{z}$\n$\\sim \\mathcal{N}(0,I)$": "#fff2cc",
}

for label, (x, y, w, h) in boxes.items():
    fc = colors.get(label, "#ffffff")
    rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="black", linewidth=1.2, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=8, zorder=3)

# Arrows
arrow_style = dict(arrowstyle="->", color="black", linewidth=1.2)
from matplotlib.patches import FancyArrowPatch

arrows = [
    ((2.3, 5.0), (3.0, 5.0)),    # Expert -> VAE Encoder
    ((4.2, 5.0), (5.0, 5.0)),    # VAE Encoder -> Style z
    ((5.6, 4.5), (5.6, 3.5)),    # Style z -> Policy
    ((7.0, 3.0), (8.0, 3.0)),    # Policy -> Simulator
    ((8.75, 3.5), (8.75, 4.5)),  # Simulator -> Random z (feedback)
    ((8.0, 5.0), (6.2, 5.0)),    # Random z -> Style z
    ((5.0, 3.0), (3.5, 1.7)),    # Policy -> PPO
    ((5.75, 2.5), (5.75, 1.7)),  # Policy -> BC Loss
    ((2.75, 0.8), (5.0, 0.8)),   # PPO connects to BC
]

for (x1, y1), (x2, y2) in arrows:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", linewidth=1.0))

# Labels for phases
ax.text(5.0, 5.8, "Training Phase", fontsize=10, fontweight="bold", ha="center", style="italic")
ax.text(1.0, 5.7, "Pre-training", fontsize=9, color=C_GREEN, ha="center")
ax.text(8.75, 5.7, "RL Training", fontsize=9, color=C_RED, ha="center")

fig.suptitle("Style-Conditioned Self-Play (SCSP) Framework", fontsize=13, fontweight="bold")
save(fig, "fig_framework")


print(f"\nAll figures saved to: {FIG_DIR}/")
print("Files:")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  {f}")
