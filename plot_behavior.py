"""
Behavior Analysis Plotting Script
===================================
Loads per-project behavior JSON results and WOSAC exp1 JSON results,
then generates comprehensive comparison figures:

  1.  Speed distribution –– violin per project
  2.  Acceleration distribution –– violin per project
  3.  Deceleration distribution –– violin per project
  4.  Max-speed vs max-decel scatter (behavioral diversity)
  5.  Goal rate –– bar chart per project
  6.  Incident rate –– bar chart per project
  7.  ADE / FDE –– bar chart per project
  8.  WOSAC num_collisions_sim + num_offroad_sim bars (from exp1 JSONs)
  9.  Jerk distribution –– violin per project (smoothness)
  10. Comprehensive summary figure (all key metrics, one page)
  11. Speed CDF overlay (all projects on one plot)
  12. Acceleration CDF overlay

Usage:
    python /users/saeani/src/guided_original/plot_behavior.py \
        --behavior-dir /users/saeani/src/guided_original/behavior_results \
        --wosac-dir    /data/lynx/saeani/evaluation_results/exp1 \
        --output-dir   /users/saeani/src/guided_original/behavior_figures
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# ─────────────────────────────────────────────────────────────────────────────
# Project ordering and display names
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ORDER = [
    "guided_full",
    "guided_full_dropout",
    "guided_reward",
    "hetero_v1",
    "hetero_v2",
    "vae_hybrid",
    "vae_hybrid_ll0",
    "vae_hybrid_noguide",
    "causal_encoder",
    "causal_encoder_ll0",
    "vae_1d_ll0",
    "vae_guided_bc",
    "vae_guided_nobc",
    "vae_randomz_bc",
    "vae_1d",
    "causal_guided_bc",
    "causal_guided_nobc",
    "causal_noguided_bc",
    "hetero_v1_guided_bc",
    "hetero_v1_guided_nobc",
    "hetero_v1_noguided_bc",
    "vae_1d_guided_bc",
    "vae_1d_guided_nobc",
    "vae_1d_randomz_bc",
    "hetero_v2_guided_bc",
    "hetero_v2_guided_nobc",
    "hetero_v2_noguided_bc",
]

DISPLAY_NAMES = {
    "guided_full":          "Guided\nFull",
    "guided_full_dropout":  "Guided\nDropout",
    "guided_reward":        "Guided\nReward",
    "hetero_v1":            "Hetero\nv1",
    "hetero_v2":            "Hetero\nv2",
    "vae_hybrid":           "VAE-4D\nLL=1",
    "vae_hybrid_ll0":       "VAE-4D\nLL=0",
    "vae_hybrid_noguide":   "VAE-4D\nNo-Guide",
    "causal_encoder":       "Causal\nLL=1",
    "causal_encoder_ll0":   "Causal\nLL=0",
    "vae_1d_ll0":           "VAE-1D\nLL=0",
    "vae_guided_bc":        "VAE\nGuided-BC",
    "vae_guided_nobc":      "VAE\nGuided-noBC",
    "vae_randomz_bc":       "VAE\nRandZ-BC",
    "vae_1d":               "VAE\n1D",
    "causal_guided_bc":     "Causal\nGuided-BC",
    "causal_guided_nobc":   "Causal\nGuided-noBC",
    "causal_noguided_bc":   "Causal\nnoGuided-BC",
    "hetero_v1_guided_bc":    "H1\nGuided-BC",
    "hetero_v1_guided_nobc":  "H1\nGuided-noBC",
    "hetero_v1_noguided_bc":  "H1\nnoGuided-BC",
    "vae_1d_guided_bc":       "V1D\nGuided-BC",
    "vae_1d_guided_nobc":     "V1D\nGuided-noBC",
    "vae_1d_randomz_bc":      "V1D\nRandZ-BC",
    "hetero_v2_guided_bc":    "H2\nGuided-BC",
    "hetero_v2_guided_nobc":  "H2\nGuided-noBC",
    "hetero_v2_noguided_bc":  "H2\nnoGuided-BC",
}

# Group colours
GROUP_COLORS = {
    "guided_full":          "#4878CF",
    "guided_full_dropout":  "#4878CF",
    "guided_reward":        "#4878CF",
    "hetero_v1":            "#6ACC65",
    "hetero_v2":            "#6ACC65",
    "vae_hybrid":           "#D65F5F",
    "vae_hybrid_ll0":       "#D65F5F",
    "vae_hybrid_noguide":   "#D65F5F",
    "causal_encoder":       "#B47CC7",
    "causal_encoder_ll0":   "#B47CC7",
    "vae_1d_ll0":           "#D65F5F",
    "vae_guided_bc":        "#E6550D",
    "vae_guided_nobc":      "#FDAE6B",
    "vae_randomz_bc":       "#3182BD",
    "vae_1d":               "#31A354",
    "causal_guided_bc":     "#9467BD",
    "causal_guided_nobc":   "#C5B0D5",
    "causal_noguided_bc":   "#8C564B",
    "hetero_v1_guided_bc":    "#6ACC65",
    "hetero_v1_guided_nobc":  "#6ACC65",
    "hetero_v1_noguided_bc":  "#6ACC65",
    "vae_1d_guided_bc":       "#D65F5F",
    "vae_1d_guided_nobc":     "#D65F5F",
    "vae_1d_randomz_bc":      "#D65F5F",
    "hetero_v2_guided_bc":    "#6ACC65",
    "hetero_v2_guided_nobc":  "#6ACC65",
    "hetero_v2_noguided_bc":  "#6ACC65",
}

WOSAC_TO_RUN = {
    "pufferdrive_guided_full":          "guided_full",
    "pufferdrive_guided_full_dropout":  "guided_full_dropout",
    "pufferdrive_guided_reward":        "guided_reward",
    "pufferdrive_hetero_v1":            "hetero_v1",
    "pufferdrive_hetero_v2":            "hetero_v2",
    "pufferdrive_vae_hybrid":           "vae_hybrid",
    "pufferdrive_vae_hybrid_ll0":       "vae_hybrid_ll0",
    "pufferdrive_vae_hybrid_noguide":   "vae_hybrid_noguide",
    "pufferdrive_causal_encoder":       "causal_encoder",
    "pufferdrive_causal_encoder_ll0":   "causal_encoder_ll0",
    "pufferdrive_vae_1d_ll0":           "vae_1d_ll0",
    "pufferdrive_vae_guided_bc":        "vae_guided_bc",
    "pufferdrive_vae_guided_nobc":      "vae_guided_nobc",
    "pufferdrive_vae_randomz_bc":       "vae_randomz_bc",
    "pufferdrive_vae_1d":               "vae_1d",
    "pufferdrive_causal_guided_bc":     "causal_guided_bc",
    "pufferdrive_causal_guided_nobc":   "causal_guided_nobc",
    "pufferdrive_causal_noguided_bc":   "causal_noguided_bc",
    "pufferdrive_hetero_v1_guided_bc":    "hetero_v1_guided_bc",
    "pufferdrive_hetero_v1_guided_nobc":  "hetero_v1_guided_nobc",
    "pufferdrive_hetero_v1_noguided_bc":  "hetero_v1_noguided_bc",
    "pufferdrive_vae_1d_guided_bc":       "vae_1d_guided_bc",
    "pufferdrive_vae_1d_guided_nobc":     "vae_1d_guided_nobc",
    "pufferdrive_vae_1d_randomz_bc":      "vae_1d_randomz_bc",
    "pufferdrive_hetero_v2_guided_bc":    "hetero_v2_guided_bc",
    "pufferdrive_hetero_v2_guided_nobc":  "hetero_v2_guided_nobc",
    "pufferdrive_hetero_v2_noguided_bc":  "hetero_v2_noguided_bc",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_behavior(behavior_dir):
    """Load per-project behavior JSONs → dict keyed by run_name."""
    data = {}
    for f in glob.glob(os.path.join(behavior_dir, "*_behavior.json")):
        with open(f) as fh:
            d = json.load(fh)
        run_name = d.get("_run_name", "")
        if run_name:
            data[run_name] = d
    return data


def load_wosac(wosac_dir):
    """Load WOSAC exp1 results → dict keyed by short run_name."""
    data = {}
    if not wosac_dir or not os.path.isdir(wosac_dir):
        return data
    for f in sorted(glob.glob(os.path.join(wosac_dir, "*.json"))):
        fname = os.path.basename(f)
        if "_sweep" in fname:
            continue
        with open(f) as fh:
            d = json.load(fh)
        project = d.get("_project", "")
        run_name = WOSAC_TO_RUN.get(project, "")
        if run_name:
            data[run_name] = d  # latest file wins (glob sorted)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Helper: extract per-agent distribution arrays for one metric
# ─────────────────────────────────────────────────────────────────────────────
def get_per_agent_data(bdata, metric_key, project_order):
    """Return list of numpy arrays (one per project in order)."""
    result = []
    for pname in project_order:
        if pname not in bdata:
            result.append(np.array([]))
            continue
        agg = bdata[pname].get("aggregate", {})
        stat = agg.get(metric_key, {})
        # Reconstruct approximate distribution from percentile stats
        # Use all scenario per-agent values instead when available
        vals = []
        for sc in bdata[pname].get("scenarios", []):
            vals.extend(sc.get(
                metric_key.replace("speed_mean_per_agent", "speed_mean_per_agent")
                .replace("max_accel_per_agent", "max_accel_per_agent"), []
            ))
        if vals:
            result.append(np.array(vals, dtype=np.float32))
        else:
            result.append(np.array([]))
    return result


def get_per_agent_raw(bdata, field, project_order):
    """Extract raw per-agent arrays across all scenarios for a given field."""
    result = []
    for pname in project_order:
        if pname not in bdata:
            result.append(np.array([]))
            continue
        vals = []
        for sc in bdata[pname].get("scenarios", []):
            v = sc.get(field, [])
            if isinstance(v, list):
                finite = [x for x in v if np.isfinite(x)]
                vals.extend(finite)
        result.append(np.array(vals, dtype=np.float32) if vals else np.array([]))
    return result


def get_step_data(bdata, field, project_order):
    """Extract step-level samples for a given field (speed_all_steps, etc.)."""
    result = []
    for pname in project_order:
        if pname not in bdata:
            result.append(np.array([]))
            continue
        vals = []
        for sc in bdata[pname].get("scenarios", []):
            v = sc.get(field, [])
            if isinstance(v, list):
                finite = [x for x in v if np.isfinite(x) and abs(x) < 1e4]
                vals.extend(finite)
        result.append(np.array(vals, dtype=np.float32) if vals else np.array([]))
    return result


def get_scalar(bdata, project_order, field, from_agg_key=None, from_scenario=False):
    """Return dict mapping run_name → scalar value."""
    out = {}
    for pname in project_order:
        if pname not in bdata:
            continue
        if from_scenario:
            vals = [sc.get(field, float("nan"))
                    for sc in bdata[pname].get("scenarios", [])
                    if np.isfinite(sc.get(field, float("nan")))]
            out[pname] = float(np.mean(vals)) if vals else float("nan")
        else:
            agg_key = from_agg_key or f"scenario_mean_{field}"
            out[pname] = bdata[pname].get("aggregate", {}).get(agg_key, float("nan"))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Styling helpers
# ─────────────────────────────────────────────────────────────────────────────
def apply_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })


def violin_plot(ax, data_list, labels, colors, ylabel, title,
                clip_lo=None, clip_hi=None, showmeans=True):
    """Draw a violin plot with individual colours per violin."""
    positions = list(range(1, len(labels) + 1))
    filtered = []
    valid_positions = []
    valid_labels = []
    valid_colors = []

    for i, (d, lab, col) in enumerate(zip(data_list, labels, colors)):
        d = np.asarray(d)
        if clip_lo is not None:
            d = d[d >= clip_lo]
        if clip_hi is not None:
            d = d[d <= clip_hi]
        if len(d) < 3:
            continue
        filtered.append(d)
        valid_positions.append(positions[i])
        valid_labels.append(lab)
        valid_colors.append(col)

    if not filtered:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    parts = ax.violinplot(filtered, positions=valid_positions,
                          showmeans=showmeans, showmedians=True,
                          showextrema=True)

    # Colour each violin body
    for body, col in zip(parts["bodies"], valid_colors):
        body.set_facecolor(col)
        body.set_alpha(0.75)
        body.set_edgecolor("k")
        body.set_linewidth(0.5)

    for part_name in ("cmeans", "cmedians", "cmins", "cmaxes", "cbars"):
        if part_name in parts:
            parts[part_name].set_color("black")
            parts[part_name].set_linewidth(1.0)

    ax.set_xticks(valid_positions)
    ax.set_xticklabels(valid_labels, rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def bar_chart(ax, values_dict, project_order, ylabel, title,
              fmt=".3f", color_override=None):
    """Bar chart from dict {run_name: value}."""
    x, heights, colors, labels = [], [], [], []
    for pname in project_order:
        v = values_dict.get(pname, float("nan"))
        if np.isfinite(v):
            x.append(pname)
            heights.append(v)
            colors.append(color_override or GROUP_COLORS.get(pname, "#888888"))
            labels.append(DISPLAY_NAMES.get(pname, pname))

    if not heights:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    positions = list(range(len(x)))
    bars = ax.bar(positions, heights, color=[color_override or GROUP_COLORS.get(n, "#888888") for n in x],
                  edgecolor="black", linewidth=0.5, alpha=0.85)

    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, h + max(heights) * 0.01,
                f"{h:{fmt}}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def cdf_plot(ax, data_list, labels, colors, xlabel, title,
             clip_lo=None, clip_hi=None):
    """Overlay CDFs for all projects."""
    for d, lab, col in zip(data_list, labels, colors):
        d = np.asarray(d)
        if clip_lo is not None:
            d = d[d >= clip_lo]
        if clip_hi is not None:
            d = d[d <= clip_hi]
        if len(d) < 3:
            continue
        xs = np.sort(d)
        ys = np.linspace(0, 1, len(xs))
        ax.step(xs, ys, where="post", label=lab, color=col, linewidth=1.5, alpha=0.85)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.7)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Build group legend patches
# ─────────────────────────────────────────────────────────────────────────────
def group_legend():
    return [
        mpatches.Patch(color="#4878CF", label="Guided (A)"),
        mpatches.Patch(color="#6ACC65", label="Hetero (B)"),
        mpatches.Patch(color="#D65F5F", label="VAE  (C)"),
        mpatches.Patch(color="#B47CC7", label="Causal (D)"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Figure generation
# ─────────────────────────────────────────────────────────────────────────────
def fig_speed_violin(bdata, projects, out_dir):
    fig, ax = plt.subplots(figsize=(14, 5))
    data   = get_step_data(bdata, "speed_all_steps", projects)
    labels = [DISPLAY_NAMES.get(p, p) for p in projects]
    colors = [GROUP_COLORS.get(p, "#888888") for p in projects]
    violin_plot(ax, data, labels, colors,
                ylabel="Speed (m/s)",
                title="Speed Distribution per Project (all agents, all steps)",
                clip_lo=0.0, clip_hi=30.0)
    fig.legend(handles=group_legend(), loc="upper right", fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, "fig_speed_violin")


def fig_accel_violin(bdata, projects, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    step_accel = get_step_data(bdata, "accel_all_steps", projects)
    labels = [DISPLAY_NAMES.get(p, p) for p in projects]
    colors = [GROUP_COLORS.get(p, "#888888") for p in projects]

    # Acceleration (positive)
    pos_data = [d[d >= 0] for d in step_accel]
    violin_plot(axes[0], pos_data, labels, colors,
                ylabel="Acceleration (m/s²)",
                title="Positive Acceleration Distribution",
                clip_lo=0.0, clip_hi=8.0)

    # Deceleration (absolute value of negatives)
    neg_data = [np.abs(d[d < 0]) for d in step_accel]
    violin_plot(axes[1], neg_data, labels, colors,
                ylabel="Deceleration magnitude (m/s²)",
                title="Deceleration Distribution",
                clip_lo=0.0, clip_hi=8.0)

    fig.suptitle("Acceleration / Deceleration Distributions", fontsize=13, y=1.02)
    fig.legend(handles=group_legend(), loc="upper right", fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, "fig_accel_violin")


def fig_max_accel_decel_bars(bdata, projects, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    max_accel = get_per_agent_raw(bdata, "max_accel_per_agent", projects)
    max_decel = get_per_agent_raw(bdata, "max_decel_per_agent", projects)
    labels = [DISPLAY_NAMES.get(p, p) for p in projects]
    colors = [GROUP_COLORS.get(p, "#888888") for p in projects]

    violin_plot(axes[0], max_accel, labels, colors,
                ylabel="Max acceleration per agent (m/s²)",
                title="Maximum Acceleration per Agent",
                clip_lo=0.0, clip_hi=10.0)
    violin_plot(axes[1], max_decel, labels, colors,
                ylabel="Max deceleration per agent (m/s²)",
                title="Maximum Deceleration per Agent",
                clip_lo=0.0, clip_hi=10.0)

    fig.suptitle("Peak Acceleration/Deceleration per Agent", fontsize=13, y=1.02)
    fig.legend(handles=group_legend(), loc="upper right", fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, "fig_max_accel_decel")


def fig_speed_cdf(bdata, projects, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    data   = get_step_data(bdata, "speed_all_steps", projects)
    labels = [DISPLAY_NAMES.get(p, p).replace("\n", " ") for p in projects]
    colors = [GROUP_COLORS.get(p, "#888888") for p in projects]
    cdf_plot(ax, data, labels, colors,
             xlabel="Speed (m/s)", title="Speed CDF – all Projects",
             clip_lo=0.0, clip_hi=30.0)
    plt.tight_layout()
    _save(fig, out_dir, "fig_speed_cdf")


def fig_accel_cdf(bdata, projects, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    step_accel = get_step_data(bdata, "accel_all_steps", projects)
    labels = [DISPLAY_NAMES.get(p, p).replace("\n", " ") for p in projects]
    colors = [GROUP_COLORS.get(p, "#888888") for p in projects]

    pos_data = [d[d >= 0] for d in step_accel]
    neg_data = [np.abs(d[d < 0]) for d in step_accel]
    cdf_plot(axes[0], pos_data, labels, colors,
             xlabel="Acceleration (m/s²)", title="Acceleration CDF",
             clip_lo=0.0, clip_hi=8.0)
    cdf_plot(axes[1], neg_data, labels, colors,
             xlabel="Deceleration magnitude (m/s²)", title="Deceleration CDF",
             clip_lo=0.0, clip_hi=8.0)

    plt.tight_layout()
    _save(fig, out_dir, "fig_accel_cdf")


def fig_goal_rate(bdata, projects, out_dir):
    vals = {}
    for p in projects:
        if p not in bdata:
            continue
        agg = bdata[p].get("aggregate", {})
        vals[p] = agg.get("overall_goal_rate", float("nan"))

    fig, ax = plt.subplots(figsize=(12, 4))
    bar_chart(ax, vals, projects,
              ylabel="Goal Achievement Rate", title="Goal Achievement Rate per Project",
              fmt=".3f")
    ax.set_ylim(0, 1.1)
    fig.legend(handles=group_legend(), loc="upper right", fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, "fig_goal_rate")


def fig_incident_rate(bdata, projects, out_dir):
    vals = {}
    for p in projects:
        if p not in bdata:
            continue
        agg = bdata[p].get("aggregate", {})
        vals[p] = agg.get("overall_incident_rate", float("nan"))

    fig, ax = plt.subplots(figsize=(12, 4))
    bar_chart(ax, vals, projects,
              ylabel="Incident Rate (collision/off-road)",
              title="Incident Rate per Project  (reward < −0.3)",
              fmt=".3f", color_override=None)
    fig.legend(handles=group_legend(), loc="upper right", fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, "fig_incident_rate")


def fig_ade_fde(bdata, projects, out_dir):
    ade_vals, fde_vals = {}, {}
    for p in projects:
        if p not in bdata:
            continue
        agg = bdata[p].get("aggregate", {})
        ade_stat = agg.get("ade_per_agent", {})
        fde_stat = agg.get("fde_per_agent", {})
        ade_vals[p] = ade_stat.get("mean", float("nan"))
        fde_vals[p] = fde_stat.get("mean", float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bar_chart(axes[0], ade_vals, projects,
              ylabel="ADE (m)", title="Average Displacement Error (ADE)", fmt=".2f")
    bar_chart(axes[1], fde_vals, projects,
              ylabel="FDE (m)", title="Final Displacement Error (FDE)", fmt=".2f")
    fig.legend(handles=group_legend(), loc="upper right", fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, "fig_ade_fde")


def fig_wosac_collision_offroad(wosac, projects, out_dir):
    """Bar chart from WOSAC exp1 data: num_collisions_sim, num_offroad_sim."""
    if not wosac:
        return
    col_vals, off_vals = {}, {}
    for p in projects:
        if p not in wosac:
            continue
        col_vals[p] = wosac[p].get("num_collisions_sim", float("nan"))
        off_vals[p] = wosac[p].get("num_offroad_sim", float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bar_chart(axes[0], col_vals, projects,
              ylabel="Collisions / agent",
              title="WOSAC Collision Rate (Simulated)\n[from WOSAC exp1]", fmt=".4f")
    bar_chart(axes[1], off_vals, projects,
              ylabel="Off-road events / agent",
              title="WOSAC Off-road Rate (Simulated)\n[from WOSAC exp1]", fmt=".4f")
    fig.legend(handles=group_legend(), loc="upper right", fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, "fig_wosac_collision_offroad")


def fig_jerk_violin(bdata, projects, out_dir):
    fig, ax = plt.subplots(figsize=(14, 5))
    data   = get_step_data(bdata, "jerk_all_steps", projects)
    # Use absolute jerk
    abs_data = [np.abs(d) for d in data]
    labels = [DISPLAY_NAMES.get(p, p) for p in projects]
    colors = [GROUP_COLORS.get(p, "#888888") for p in projects]
    violin_plot(ax, abs_data, labels, colors,
                ylabel="|Jerk| (m/s³)",
                title="Jerk Distribution per Project  (lower = smoother driving)",
                clip_lo=0.0, clip_hi=20.0)
    fig.legend(handles=group_legend(), loc="upper right", fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, "fig_jerk_violin")


def fig_speed_scatter(bdata, projects, out_dir):
    """Max-speed vs max-decel scatter: shows driver style diversity."""
    fig, ax = plt.subplots(figsize=(9, 7))
    for p in projects:
        if p not in bdata:
            continue
        ag = bdata[p].get("aggregate", {})
        spd  = ag.get("speed_max_per_agent", {})
        decel= ag.get("max_decel_per_agent", {})
        if not spd or not decel:
            continue
        # draw mean ± std as error cross
        ax.errorbar(
            spd.get("mean", float("nan")), decel.get("mean", float("nan")),
            xerr=spd.get("std",  0.0),    yerr=decel.get("std",  0.0),
            fmt="o", color=GROUP_COLORS.get(p, "#888888"),
            markersize=9, capsize=4, elinewidth=1.5, linewidth=2,
            label=DISPLAY_NAMES.get(p, p).replace("\n", " "),
        )
    ax.set_xlabel("Max Speed per Agent – Mean (m/s)")
    ax.set_ylabel("Max Deceleration per Agent – Mean (m/s²)")
    ax.set_title("Driving Style Space: Max Speed vs Max Deceleration\n(mean ± 1 std across agents)")
    ax.legend(fontsize=8, loc="best", framealpha=0.75)
    ax.grid(linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig_speed_decel_scatter")


def fig_wosac_realism(wosac, projects, out_dir):
    """Bar chart of WOSAC realism meta score for reference."""
    if not wosac:
        return
    vals = {p: wosac[p].get("realism_meta_score", float("nan"))
            for p in projects if p in wosac}
    fig, ax = plt.subplots(figsize=(12, 4))
    bar_chart(ax, vals, projects,
              ylabel="Realism Meta Score",
              title="WOSAC Realism Meta Score  (from exp1)", fmt=".4f")
    ax.set_ylim(0, 1.0)
    fig.legend(handles=group_legend(), loc="upper right", fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, "fig_wosac_realism")


def fig_comprehensive_summary(bdata, wosac, projects, out_dir):
    """One-page 3×3 summary of all key metrics."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.ravel()

    labels    = [DISPLAY_NAMES.get(p, p) for p in projects]
    colors    = [GROUP_COLORS.get(p, "#888888") for p in projects]
    positions = list(range(len(projects)))

    def _bar(ax, vals_dict, title, ylabel, fmt=".3f", ylim=None):
        hs, cs, ls, ps = [], [], [], []
        for i, p in enumerate(projects):
            v = vals_dict.get(p, float("nan"))
            if np.isfinite(v):
                hs.append(v); cs.append(colors[i])
                ls.append(labels[i]); ps.append(i)
        if hs:
            bars = ax.bar(ps, hs, color=cs, edgecolor="black", linewidth=0.4,
                          alpha=0.85)
            for bar, h in zip(bars, hs):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + max(hs) * 0.01,
                        f"{h:{fmt}}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ylim:
            ax.set_ylim(*ylim)

    # 1. WOSAC Realism
    wosac_realism = {p: wosac[p].get("realism_meta_score", float("nan"))
                     for p in projects if p in wosac}
    _bar(axes[0], wosac_realism, "WOSAC Realism Score", "Score", fmt=".3f", ylim=(0, 1))

    # 2. Goal Rate
    goal_rate = {p: bdata[p].get("aggregate", {}).get("overall_goal_rate", float("nan"))
                 for p in projects if p in bdata}
    _bar(axes[1], goal_rate, "Goal Achievement Rate", "Rate", fmt=".3f", ylim=(0, 1))

    # 3. Incident Rate
    inc_rate = {p: bdata[p].get("aggregate", {}).get("overall_incident_rate", float("nan"))
                for p in projects if p in bdata}
    _bar(axes[2], inc_rate, "Incident Rate (reward-based)", "Rate", fmt=".3f")

    # 4. WOSAC Collisions
    col_wosac = {p: wosac[p].get("num_collisions_sim", float("nan"))
                 for p in projects if p in wosac}
    _bar(axes[3], col_wosac, "WOSAC Collision Rate (sim)", "Collisions/agent", fmt=".4f")

    # 5. WOSAC Off-road
    offrd_wosac = {p: wosac[p].get("num_offroad_sim", float("nan"))
                   for p in projects if p in wosac}
    _bar(axes[4], offrd_wosac, "WOSAC Off-road Rate (sim)", "Events/agent", fmt=".4f")

    # 6. ADE
    ade_vals = {p: bdata[p].get("aggregate", {}).get("ade_per_agent", {}).get("mean", float("nan"))
                for p in projects if p in bdata}
    _bar(axes[5], ade_vals, "ADE (m)", "m", fmt=".2f")

    # 7. Mean Speed
    spd_vals = {p: bdata[p].get("aggregate", {}).get(
                    "speed_mean_per_agent", {}).get("mean", float("nan"))
                for p in projects if p in bdata}
    _bar(axes[6], spd_vals, "Mean Speed (m/s)", "m/s", fmt=".1f")

    # 8. Mean Max Decel
    dec_vals = {p: bdata[p].get("aggregate", {}).get(
                    "max_decel_per_agent", {}).get("mean", float("nan"))
                for p in projects if p in bdata}
    _bar(axes[7], dec_vals, "Mean Max Decel (m/s²)", "m/s²", fmt=".2f")

    # 9. Mean Jerk
    jerk_vals = {p: bdata[p].get("aggregate", {}).get(
                     "mean_abs_jerk_per_agent", {}).get("mean", float("nan"))
                 for p in projects if p in bdata}
    _bar(axes[8], jerk_vals, "Mean |Jerk| (m/s³)", "m/s³", fmt=".2f")

    fig.suptitle("Comprehensive Behavioral Analysis – All Projects", fontsize=14, y=1.002)
    fig.legend(handles=group_legend(), loc="upper right", fontsize=9,
               bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    _save(fig, out_dir, "fig_comprehensive_summary")


def fig_wosac_kinematic_breakdown(wosac, projects, out_dir):
    """WOSAC kinematic likelihood metrics side by side."""
    if not wosac:
        return
    metrics = [
        ("likelihood_linear_speed",        "LL Speed"),
        ("likelihood_linear_acceleration", "LL Accel"),
        ("likelihood_angular_speed",       "LL Ang Spd"),
        ("kinematic_metrics",              "Kinematic"),
        ("interactive_metrics",            "Interactive"),
        ("map_based_metrics",              "Map-Based"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    for ax, (key, title) in zip(axes, metrics):
        vals = {p: wosac[p].get(key, float("nan")) for p in projects if p in wosac}
        bar_chart(ax, vals, projects, ylabel="Score", title=title, fmt=".3f")
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", labelsize=7)

    fig.suptitle("WOSAC Metric Breakdown  (from exp1)", fontsize=13)
    fig.legend(handles=group_legend(), loc="upper right", fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, "fig_wosac_metric_breakdown")


def fig_per_split_comparison(bdata, projects, out_dir):
    """
    Compare train vs val goal_rate and incident_rate side-by-side per project.
    """
    train_goal, val_goal = {}, {}
    train_inc,  val_inc  = {}, {}

    for p in projects:
        if p not in bdata:
            continue
        train_scs = [sc for sc in bdata[p].get("scenarios", []) if sc.get("split") == "train"]
        val_scs   = [sc for sc in bdata[p].get("scenarios", []) if sc.get("split") == "val"]

        def _rate(scs, key):
            agents = sum(len(sc.get(key + "_per_agent", [])) for sc in scs)
            hits   = sum(sum(1 for v in sc.get(key + "_per_agent", []) if v)
                         for sc in scs)
            return float(hits / agents) if agents > 0 else float("nan")

        train_goal[p] = _rate(train_scs, "goal_achieved")
        val_goal[p]   = _rate(val_scs,   "goal_achieved")
        train_inc[p]  = _rate(train_scs, "incident")
        val_inc[p]    = _rate(val_scs,   "incident")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    labels    = [DISPLAY_NAMES.get(p, p) for p in projects]
    positions = np.arange(len(projects))
    w = 0.35

    for ax, (t_vals, v_vals, title, ylabel) in zip(
        axes,
        [
            (train_goal, val_goal, "Goal Rate: Train vs Val", "Goal Rate"),
            (train_inc,  val_inc,  "Incident Rate: Train vs Val", "Incident Rate"),
        ]
    ):
        t_heights = [t_vals.get(p, float("nan")) for p in projects]
        v_heights = [v_vals.get(p, float("nan")) for p in projects]
        b1 = ax.bar(positions - w/2,
                    [h if np.isfinite(h) else 0 for h in t_heights],
                    w, label="Train", color="#5588CC", alpha=0.8, edgecolor="black", lw=0.4)
        b2 = ax.bar(positions + w/2,
                    [h if np.isfinite(h) else 0 for h in v_heights],
                    w, label="Val",   color="#EE8833", alpha=0.8, edgecolor="black", lw=0.4)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save(fig, out_dir, "fig_train_val_split")


# ─────────────────────────────────────────────────────────────────────────────
# Save helper
# ─────────────────────────────────────────────────────────────────────────────
def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {name}.png/.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# CLI and main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--behavior-dir", required=True,
                   help="Dir containing *_behavior.json files")
    p.add_argument("--wosac-dir",    default="",
                   help="Dir containing WOSAC exp1 *.json files (optional)")
    p.add_argument("--output-dir",   required=True,
                   help="Where to save figures")
    p.add_argument("--projects",     default="",
                   help="Comma-separated list of projects to include "
                        "(default: all found in behavior_dir)")
    return p.parse_args()


def main():
    args = parse_args()
    apply_style()

    bdata = load_behavior(args.behavior_dir)
    wosac = load_wosac(args.wosac_dir)

    if not bdata:
        print(f"ERROR: No behavior JSON files found in {args.behavior_dir}")
        return

    if args.projects:
        projects = [p.strip() for p in args.projects.split(",")]
    else:
        # Use canonical order, keeping only projects with data
        projects = [p for p in PROJECT_ORDER if p in bdata]
        # Also add any other projects found
        for p in bdata:
            if p not in projects:
                projects.append(p)

    print(f"Projects with behavior data : {projects}")
    print(f"Projects with WOSAC data    : {list(wosac.keys())}")
    print(f"Output dir                  : {args.output_dir}")
    print()

    print("Generating figures …")
    fig_speed_violin(bdata, projects, args.output_dir)
    fig_accel_violin(bdata, projects, args.output_dir)
    fig_max_accel_decel_bars(bdata, projects, args.output_dir)
    fig_speed_cdf(bdata, projects, args.output_dir)
    fig_accel_cdf(bdata, projects, args.output_dir)
    fig_goal_rate(bdata, projects, args.output_dir)
    fig_incident_rate(bdata, projects, args.output_dir)
    fig_ade_fde(bdata, projects, args.output_dir)
    fig_jerk_violin(bdata, projects, args.output_dir)
    fig_speed_scatter(bdata, projects, args.output_dir)
    fig_wosac_collision_offroad(wosac, projects, args.output_dir)
    fig_wosac_realism(wosac, projects, args.output_dir)
    fig_wosac_kinematic_breakdown(wosac, projects, args.output_dir)
    fig_comprehensive_summary(bdata, wosac, projects, args.output_dir)
    fig_per_split_comparison(bdata, projects, args.output_dir)

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Project':<25} {'Goal%':>7} {'Inc%':>7} {'MeanSpd':>9} "
          f"{'MaxSpd':>7} {'MaxDecel':>9} {'ADE':>7} {'FDE':>7}")
    print("─" * 90)
    for p in projects:
        if p not in bdata:
            continue
        agg = bdata[p].get("aggregate", {})
        goal  = agg.get("overall_goal_rate", float("nan")) * 100
        inc   = agg.get("overall_incident_rate", float("nan")) * 100
        mspd  = agg.get("speed_mean_per_agent", {}).get("mean", float("nan"))
        maxsp = agg.get("speed_max_per_agent",  {}).get("mean", float("nan"))
        dec   = agg.get("max_decel_per_agent",  {}).get("mean", float("nan"))
        ade   = agg.get("ade_per_agent",         {}).get("mean", float("nan"))
        fde   = agg.get("fde_per_agent",         {}).get("mean", float("nan"))
        print(f"  {p:<23} {goal:>6.1f}% {inc:>6.1f}%  {mspd:>8.1f}  "
              f"{maxsp:>6.1f}  {dec:>8.2f}  {ade:>6.2f}  {fde:>6.2f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
