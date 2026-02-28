#!/usr/bin/env python3
"""
Compare Training Paradigms — Integrated (Local Files Only)
==========================================================
Same analysis as ``compare_training_paradigms.py`` but reads exclusively from
the centralized ``results/training_runs/`` directory created by
``collect_training_results.py``.  **No wandb API required.**

Data sources (per paradigm subdirectory):
  • ``wandb/history.csv``       — exported metric timeseries
  • ``wandb/wandb-summary.json`` — final run summary (for step↔epoch mapping)
  • ``checkpoints/``            — model checkpoint .pt files
  • ``run_info.json``           — paradigm metadata (run ID, label, etc.)

Usage
-----
    # Default step, default results directory
    python scripts/compare_training_paradigms_integrated.py

    # Custom step
    python scripts/compare_training_paradigms_integrated.py --step 4017094656

    # Custom results directory
    python scripts/compare_training_paradigms_integrated.py \\
        --results-dir results/training_runs

    # Save to CSV
    python scripts/compare_training_paradigms_integrated.py \\
        --csv results/paradigm_comparison_step_5348261888.csv

Pipeline
--------
    1. python scripts/collect_training_results.py          # copy data locally
    2. python scripts/compare_training_paradigms_integrated.py  # analyze
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from typing import Any

# Try to import pandas for faster history reading; fall back to csv module
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ── Paradigm subdirectories (same order as the collect script) ───────────────
PARADIGMS: list[tuple[str, str]] = [
    ("adversarial",   "Adversarial"),
    ("guided_reward", "Guided Reward"),
    ("pure_self_play", "Pure Self-Play"),
]

# ── Metrics (same as the wandb-backed script, for consistency) ───────────────
#
# WOSAC vs Environment vs Evaluation — what's the difference?
#
#   • WOSAC Realism  — Evaluated periodically via the Waymo Open Sim Agents
#     Challenge (WOSAC) protocol.  The policy is tested on held-out WOSAC
#     scenarios and its *distributional* behavior is compared against real
#     human driving logs.  These scores measure how statistically realistic
#     the agent's driving is (kinematic plausibility, interaction quality,
#     map compliance, collision/TTC likelihoods).  This is an offline eval
#     that runs on a fixed scenario set.
#
#   • Evaluation (eval/sp_*, eval/hr_*, eval/Δ_*) — Periodic in-training
#     evaluations.  "sp" = self-play eval (all agents use the learned policy),
#     "hr" = human-replay eval (ego uses the policy, others replay logs).
#     Δ metrics are the difference between sp and hr evals.  These give a
#     snapshot of how the policy performs in controlled evaluation episodes.
#
#   • Environment (environment/*) — Rolling statistics collected during
#     *training* episodes (all 1024 agents running the current policy).
#     These reflect the training distribution: collision rate, offroad rate,
#     goal completion, etc. as seen by the learner.  Because training has
#     exploration noise and curriculum effects, these can differ from the
#     cleaner evaluation numbers.
#
METRIC_GROUPS: dict[str, dict[str, str]] = {
    "WOSAC Realism": {
        "eval/wosac_realism_meta_score_mean": "Meta Score (mean)",
        "eval/wosac_realism_meta_score_std":  "Meta Score (std)",
        "eval/wosac_kinematic_metrics":       "Kinematic Metrics",
        "eval/wosac_interactive_metrics":     "Interactive Metrics",
        "eval/wosac_map_based_metrics":       "Map-Based Metrics",
        "eval/wosac_likelihood_collision":    "Likelihood Collision",
        "eval/wosac_likelihood_ttc":          "Likelihood TTC",
        "eval/wosac_likelihood_dist_to_no":   "Likelihood Dist-to-NO",
        "eval/wosac_ade":                     "ADE",
        "eval/wosac_min_ade":                 "Min ADE",
        "eval/wosac_total_num_agents":        "Total Num Agents",
        "eval/num_collisions_sim":            "Num Collisions (sim)",
        "eval/num_collisions_ref":            "Num Collisions (ref)",
    },
    "Evaluation": {
        "eval/sp_score":            "Self-Play Score",
        "eval/sp_collision_rate":   "Self-Play Collision Rate",
        "eval/hr_score":            "Human-Replay Score",
        "eval/hr_collision_rate":   "Human-Replay Collision Rate",
        "eval/human_log_prob":      "Human Log Probability",
        "eval/Δ_score":             "Δ Score",
        "eval/Δ_cr":                "Δ Collision Rate",
        "eval/Δ_or":                "Δ Offroad Rate",
        "eval/Δ_comp":              "Δ Completion",
    },
    "Environment": {
        "environment/collision_rate":               "Collision Rate",
        "environment/collisions_per_agent":         "Collisions/Agent",
        "environment/offroad_rate":                 "Offroad Rate",
        "environment/offroad_per_agent":            "Offroad/Agent",
        "environment/completion_rate":              "Completion Rate",
        "environment/goals_reached_this_episode":   "Goals Reached/Episode",
        "environment/score":                        "Score",
        "environment/episode_return":               "Episode Return",
        "environment/dnf_rate":                     "DNF Rate",
        "environment/lane_alignment_rate":          "Lane Alignment Rate",
        "environment/speed_at_goal":                "Speed at Goal",
        "environment/perturbed_agent_count":        "Perturbed Agent Count",
        "environment/perturbed_collision_count":    "Perturbed Collision Count",
        "environment/unperturbed_collision_count":  "Unperturbed Collision Count",
    },
    "Losses": {
        "losses/policy_loss":        "Policy Loss",
        "losses/value_loss":         "Value Loss",
        "losses/entropy":            "Entropy",
        "losses/approx_kl":          "Approx KL",
        "losses/old_approx_kl":      "Old Approx KL",
        "losses/clipfrac":           "Clip Fraction",
        "losses/explained_variance": "Explained Variance",
        "losses/importance":         "Importance",
        "losses/human_nll":          "Human NLL",
    },
    "Training": {
        "agent_steps":    "Agent Steps",
        "epoch":          "Epoch (update)",
        "learning_rate":  "Learning Rate",
        "SPS":            "Steps Per Second",
    },
}

ALL_METRIC_KEYS: list[str] = [
    key for group in METRIC_GROUPS.values() for key in group
]


# ── History reading ──────────────────────────────────────────────────────────

def read_history_pandas(csv_path: str) -> "pd.DataFrame":
    """Read history CSV using pandas (fast)."""
    df = pd.read_csv(csv_path, low_memory=False)
    return df


def read_history_stdlib(csv_path: str) -> list[dict[str, Any]]:
    """Read history CSV using the stdlib csv module (no pandas needed)."""
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted: dict[str, Any] = {}
            for k, v in row.items():
                if v == "" or v is None:
                    converted[k] = None
                else:
                    try:
                        converted[k] = float(v)
                        # Promote to int if exact
                        if converted[k] == int(converted[k]) and "." not in v:
                            converted[k] = int(converted[k])
                    except (ValueError, OverflowError):
                        converted[k] = v
            rows.append(converted)
    return rows


def find_closest_row_pandas(
    df: "pd.DataFrame", target_step: int, step_col: str = "_step"
) -> dict[str, Any]:
    """Return the single row closest to *target_step* as a dict.

    This mirrors the logic in the original wandb-API script exactly:
    find ONE closest row and read ALL metrics from it.  The exported
    ``history.csv`` was created with the same ``keys`` parameter so the
    server-side downsampling matches, ensuring identical values.
    """
    if step_col not in df.columns:
        for alt in ["agent_steps", "global_step", "step"]:
            if alt in df.columns:
                step_col = alt
                break
        else:
            return {}

    df = df.dropna(subset=[step_col]).copy()
    if df.empty:
        return {}

    df["_diff"] = (df[step_col] - target_step).abs()
    idx = df["_diff"].idxmin()
    row = df.loc[idx]

    result: dict[str, Any] = {}
    for key in ["_step", step_col] + ALL_METRIC_KEYS:
        if key in row.index:
            val = row[key]
            if hasattr(val, "item"):
                val = val.item()
            if isinstance(val, float) and val != val:
                val = None
            result[key] = val
        else:
            result[key] = None

    result["_step_diff"] = int(row["_diff"])
    return result


def find_closest_row_stdlib(
    rows: list[dict[str, Any]], target_step: int, step_col: str = "_step"
) -> dict[str, Any]:
    """Return the row closest to target_step."""
    if not rows:
        return {}

    # Determine step column
    if step_col not in rows[0]:
        for alt in ["agent_steps", "global_step", "step"]:
            if alt in rows[0] and rows[0][alt] is not None:
                step_col = alt
                break
        else:
            return {}

    # Find the single closest row (mirrors the original wandb-API script)
    best_row: dict[str, Any] = {}
    best_diff = float("inf")

    for row in rows:
        step = row.get(step_col)
        if step is None:
            continue
        step = float(step)
        diff = abs(step - target_step)
        if diff < best_diff:
            best_diff = diff
            best_row = row

    if not best_row:
        return {}

    result: dict[str, Any] = {}
    for key in ["_step", step_col] + ALL_METRIC_KEYS:
        result[key] = best_row.get(key)

    result["_step_diff"] = int(best_diff)
    return result


def load_metrics_at_step(
    paradigm_dir: str, target_step: int
) -> dict[str, Any]:
    """Load metrics at *target_step* from the local history CSV."""
    history_csv = os.path.join(paradigm_dir, "wandb", "history.csv")

    if not os.path.isfile(history_csv):
        print(f"    ⚠  history.csv not found in {paradigm_dir}")
        return {}

    if HAS_PANDAS:
        df = read_history_pandas(history_csv)
        return find_closest_row_pandas(df, target_step)
    else:
        rows = read_history_stdlib(history_csv)
        return find_closest_row_stdlib(rows, target_step)


# ── Checkpoint finder ────────────────────────────────────────────────────────

def find_checkpoint_for_step(
    ckpt_dir: str, target_step: int, summary: dict[str, Any]
) -> tuple[str | None, int | None]:
    """Return (path, epoch) of the checkpoint closest to target_step."""
    total_steps = summary.get("agent_steps", summary.get("_step", 0))
    max_epoch = summary.get("epoch", 0)

    if not max_epoch or not total_steps:
        return None, None

    steps_per_epoch = total_steps / max_epoch
    target_epoch = round(target_step / steps_per_epoch)

    ckpts: list[tuple[int, str]] = []
    for path in glob.glob(os.path.join(ckpt_dir, "model_puffer_drive_*.pt")):
        m = re.search(r"model_puffer_drive_(\d+)\.pt$", path)
        if m:
            ckpts.append((int(m.group(1)), path))

    if not ckpts:
        return None, None

    ckpts.sort(key=lambda c: abs(c[0] - target_epoch))
    return ckpts[0][1], ckpts[0][0]


# ── Formatting ───────────────────────────────────────────────────────────────

def format_value(val: Any) -> str:
    """Pretty-print a metric value."""
    if val is None or (isinstance(val, float) and val != val):
        return "—"
    if isinstance(val, float):
        if abs(val) < 1e-6 and val != 0:
            return f"{val:.2e}"
        if abs(val) >= 1000:
            return f"{val:,.2f}"
        return f"{val:.6f}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare training paradigms using locally collected results "
                    "(no wandb API required).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5_348_261_888,
        help="Target training step (default: 5,348,261,888).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/training_runs",
        help="Path to the collected results directory "
             "(output of collect_training_results.py).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to save results as a CSV file.",
    )
    args = parser.parse_args()

    target_step = args.step

    # Resolve results-dir relative to the workspace
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(args.results_dir):
        results_dir = os.path.join(workspace_dir, args.results_dir)
    else:
        results_dir = args.results_dir

    if not os.path.isdir(results_dir):
        print(f"\n  ERROR: Results directory not found: {results_dir}")
        print(f"  Run  collect_training_results.py  first.\n")
        sys.exit(1)

    print(f"\n{'═' * 80}")
    print(f"  TRAINING PARADIGM COMPARISON @ step {target_step:,}")
    print(f"  (local data from {results_dir})")
    print(f"{'═' * 80}\n")

    all_results: dict[str, dict[str, Any]] = {}
    policy_paths: dict[str, str] = {}
    checkpoint_epochs: dict[str, int] = {}

    for subfolder, label in PARADIGMS:
        paradigm_dir = os.path.join(results_dir, subfolder)

        if not os.path.isdir(paradigm_dir):
            print(f"  ⚠  Directory not found: {paradigm_dir}  — skipping {label}\n")
            continue

        # Load run_info.json for metadata
        info_path = os.path.join(paradigm_dir, "run_info.json")
        run_info: dict[str, Any] = {}
        if os.path.isfile(info_path):
            with open(info_path) as f:
                run_info = json.load(f)

        run_id = run_info.get("run_id", "unknown")

        print(f"  {label}")
        print(f"    Directory : {paradigm_dir}")
        print(f"    Run ID    : {run_id}")

        # ── Load metrics from history.csv ────────────────────────────────
        metrics = load_metrics_at_step(paradigm_dir, target_step)
        actual_step = metrics.get("_step")
        step_diff = metrics.get("_step_diff", "?")
        print(f"    Actual step: {format_value(actual_step)}  "
              f"(Δ = {format_value(step_diff)} from target)")

        all_results[label] = metrics

        # ── Find checkpoint ──────────────────────────────────────────────
        ckpt_dir = os.path.join(paradigm_dir, "checkpoints")
        summary_path = os.path.join(paradigm_dir, "wandb", "wandb-summary.json")
        summary: dict[str, Any] = {}
        if os.path.isfile(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)

        if os.path.isdir(ckpt_dir):
            ckpt_path, ckpt_epoch = find_checkpoint_for_step(
                ckpt_dir, target_step, summary
            )
            if ckpt_path:
                policy_paths[label] = ckpt_path
                checkpoint_epochs[label] = ckpt_epoch
                print(f"    Checkpoint : {ckpt_path}")
                print(f"    Epoch      : {ckpt_epoch}")
            else:
                print(f"    ⚠  No checkpoints found in {ckpt_dir}")
        else:
            print(f"    ⚠  Checkpoints directory not found")

        print()

    if not all_results:
        print("No results collected.  Exiting.")
        sys.exit(1)

    # ── Comparison table ─────────────────────────────────────────────────
    labels = list(all_results.keys())
    col_width = max(22, *(len(l) for l in labels)) + 2

    def print_table_header() -> None:
        header = f"  {'Metric':<40s}"
        for label in labels:
            header += f"  {label:>{col_width}s}"
        print(header)
        print("  " + "─" * (40 + (col_width + 2) * len(labels)))

    def print_metric_row(display_name: str, key: str) -> None:
        row = f"  {display_name:<40s}"
        for label in labels:
            val = all_results[label].get(key)
            row += f"  {format_value(val):>{col_width}s}"
        print(row)

    print(f"\n{'═' * 80}")
    print(f"  COMPARISON TABLE @ step {target_step:,}")
    print(f"{'═' * 80}\n")

    for group_name, group_metrics in METRIC_GROUPS.items():
        print(f"  ┌─ {group_name} {'─' * (72 - len(group_name))}")
        print_table_header()
        for key, display_name in group_metrics.items():
            print_metric_row(display_name, key)
        print()

    # ── Policy checkpoint summary ────────────────────────────────────────
    print(f"{'═' * 80}")
    print(f"  POLICY CHECKPOINTS @ step {target_step:,}")
    print(f"{'═' * 80}\n")

    for label in labels:
        path = policy_paths.get(label)
        if path:
            print(f"  {label}:")
            print(f"    {path}")
        else:
            print(f"  {label}: ⚠  No checkpoint found")
        print()

    # ── Optional CSV export ──────────────────────────────────────────────
    if args.csv:
        csv_path = args.csv
        # Ensure directory exists
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Group", "Metric", "wandb_key"] + labels)
            for group_name, group_metrics in METRIC_GROUPS.items():
                for key, display_name in group_metrics.items():
                    row_vals = []
                    for label in labels:
                        val = all_results[label].get(key)
                        row_vals.append(val if val is not None else "")
                    writer.writerow([group_name, display_name, key] + row_vals)

            writer.writerow([])
            writer.writerow(["Policy Checkpoints"])
            writer.writerow(["Paradigm", "Checkpoint Path", "Checkpoint Epoch"])
            for label in labels:
                writer.writerow([
                    label,
                    policy_paths.get(label, "N/A"),
                    checkpoint_epochs.get(label, "N/A"),
                ])

        print(f"  ✓ Results saved to {csv_path}\n")

    print("Done.\n")


if __name__ == "__main__":
    main()
