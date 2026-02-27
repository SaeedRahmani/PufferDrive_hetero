#!/usr/bin/env python3
"""
Compare Training Paradigms — Metrics & Policy Finder
=====================================================
For a given training step, this script:

1. Retrieves metrics from Weights & Biases for the *latest* run in each of
   the three project directories (adversarial, guided reward, pure self-play).
   Metrics include WOSAC realism scores, collision/offroad/completion rates,
   loss values, and evaluation deltas.

2. Identifies the exact policy checkpoint (.pt file) corresponding to that
   step in each project, so it can be used for downstream evaluation.

Usage
-----
    # Default step (5,348,261,888):
    python scripts/compare_training_paradigms.py

    # Custom step:
    python scripts/compare_training_paradigms.py --step 4017094656

    # Specify a custom wandb entity:
    python scripts/compare_training_paradigms.py --wandb-entity my-entity

    # Specify custom project root:
    python scripts/compare_training_paradigms.py --root /users/saeani/src

    # Save output to a CSV file:
    python scripts/compare_training_paradigms.py --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path
from typing import Any

import wandb


# ── Project definitions ──────────────────────────────────────────────────────
# Each entry maps a human-readable label → directory name under --root.
PROJECT_DIRS: dict[str, str] = {
    "Adversarial":    "pufferdrive_adversarial",
    "Guided Reward":  "pufferdrive_guidedreward",
    "Pure Self-Play": "pufferdrive_puresp",
}

# ── Metrics we care about (wandb key → display name) ────────────────────────
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

# Flat list of all wandb keys we request
ALL_METRIC_KEYS: list[str] = [
    key
    for group in METRIC_GROUPS.values()
    for key in group
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def find_latest_run_id(project_dir: str) -> str:
    """Return the run ID of the latest wandb run in *project_dir*.

    Strategy (robust, avoids picking an old run):
      1. Follow the ``wandb/latest-run`` symlink created by wandb.
      2. Fall back to lexicographically sorting ``run-*`` directories
         (the timestamp prefix guarantees the latest sorts last).
    """
    wandb_dir = os.path.join(project_dir, "wandb")

    # --- strategy 1: symlink ---
    latest_link = os.path.join(wandb_dir, "latest-run")
    if os.path.islink(latest_link):
        target = os.readlink(latest_link)  # e.g. "run-20260226_163532-rc1t2i6i"
        run_id = target.rsplit("-", 1)[-1]
        return run_id

    # --- strategy 2: sort run directories ---
    run_dirs = sorted(glob.glob(os.path.join(wandb_dir, "run-*")))
    if not run_dirs:
        raise FileNotFoundError(
            f"No wandb run directories found in {wandb_dir}"
        )
    latest_dir = os.path.basename(run_dirs[-1])
    run_id = latest_dir.rsplit("-", 1)[-1]
    return run_id


def find_experiment_dir(project_dir: str, run_id: str) -> str | None:
    """Return the experiment checkpoint directory for *run_id*."""
    exp_base = os.path.join(project_dir, "experiments")
    candidate = os.path.join(exp_base, f"puffer_drive_{run_id}")
    if os.path.isdir(candidate):
        return candidate
    # Try to find any directory matching the run_id
    for d in glob.glob(os.path.join(exp_base, f"*{run_id}*")):
        if os.path.isdir(d):
            return d
    return None


def find_checkpoint_for_step(
    exp_dir: str, target_step: int, total_steps: int, max_epoch: int
) -> tuple[str | None, int | None]:
    """Return (checkpoint_path, epoch) for the checkpoint closest to *target_step*.

    Checkpoints are named ``model_puffer_drive_NNNNNN.pt`` where NNNNNN is the
    epoch (update number).  The mapping is:
        step ≈ epoch × (total_steps / max_epoch)
    """
    if max_epoch == 0 or total_steps == 0:
        return None, None

    steps_per_epoch = total_steps / max_epoch
    target_epoch = round(target_step / steps_per_epoch)

    # Gather available checkpoints
    pattern = os.path.join(exp_dir, "model_puffer_drive_*.pt")
    ckpts: list[tuple[int, str]] = []
    for path in glob.glob(pattern):
        m = re.search(r"model_puffer_drive_(\d+)\.pt$", path)
        if m:
            ckpts.append((int(m.group(1)), path))

    if not ckpts:
        return None, None

    # Find closest epoch
    ckpts.sort(key=lambda c: abs(c[0] - target_epoch))
    best_epoch, best_path = ckpts[0]
    return best_path, best_epoch


def fetch_metrics_at_step(
    run: "wandb.apis.public.Run",
    target_step: int,
    metric_keys: list[str],
) -> dict[str, Any]:
    """Fetch metrics from wandb history at (or closest to) *target_step*.

    Uses ``run.history(samples=…)`` which is server-side downsampled and fast.
    """
    keys_to_fetch = ["_step"] + metric_keys
    df = run.history(keys=keys_to_fetch, samples=10000)

    if df.empty:
        return {}

    # Find the row closest to target_step
    df = df.copy()
    df["_diff"] = (df["_step"] - target_step).abs()
    closest_idx = df["_diff"].idxmin()
    row = df.loc[closest_idx]

    result: dict[str, Any] = {}
    for key in keys_to_fetch:
        val = row.get(key)
        # Convert numpy types to native Python
        if hasattr(val, "item"):
            val = val.item()
        result[key] = val

    result["_step_diff"] = int(row["_diff"])
    return result


def format_value(val: Any) -> str:
    """Pretty-print a metric value."""
    if val is None or (isinstance(val, float) and val != val):  # NaN check
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
        description="Compare training paradigms at a specific training step.",
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
        "--root",
        type=str,
        default="/users/saeani/src",
        help="Root directory containing the three project folders.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="s-rahmani-tu-delft",
        help="Weights & Biases entity (username or team).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="pufferlib",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to save results as a CSV file.",
    )
    args = parser.parse_args()

    target_step = args.step
    root = args.root

    print(f"\n{'═' * 80}")
    print(f"  TRAINING PARADIGM COMPARISON @ step {target_step:,}")
    print(f"{'═' * 80}\n")

    api = wandb.Api()

    # Collect results for each paradigm  { label -> {metric_key: value} }
    all_results: dict[str, dict[str, Any]] = {}
    policy_paths: dict[str, str] = {}

    for label, dirname in PROJECT_DIRS.items():
        project_dir = os.path.join(root, dirname)

        if not os.path.isdir(project_dir):
            print(f"  ⚠  Directory not found: {project_dir}  — skipping {label}")
            continue

        # ── Find latest run ──────────────────────────────────────────────
        try:
            run_id = find_latest_run_id(project_dir)
        except FileNotFoundError as exc:
            print(f"  ⚠  {exc}  — skipping {label}")
            continue

        run_path = f"{args.wandb_entity}/{args.wandb_project}/{run_id}"
        print(f"  {label}")
        print(f"    Directory : {project_dir}")
        print(f"    wandb run : {run_path}")

        # ── Fetch metrics from wandb ─────────────────────────────────────
        try:
            run = api.run(run_path)
        except Exception as exc:
            print(f"    ⚠  Could not load wandb run: {exc}")
            continue

        metrics = fetch_metrics_at_step(run, target_step, ALL_METRIC_KEYS)
        actual_step = metrics.get("_step")
        step_diff = metrics.get("_step_diff", "?")
        print(f"    Actual step returned: {format_value(actual_step)}  "
              f"(Δ = {format_value(step_diff)} from target)")

        all_results[label] = metrics

        # ── Find checkpoint ──────────────────────────────────────────────
        # We need total_steps and max_epoch from the run summary to map
        # step → epoch.
        summary = dict(run.summary)
        total_steps = summary.get("agent_steps", summary.get("_step", 0))
        max_epoch = summary.get("epoch", 0)

        exp_dir = find_experiment_dir(project_dir, run_id)
        if exp_dir:
            ckpt_path, ckpt_epoch = find_checkpoint_for_step(
                exp_dir, target_step, total_steps, max_epoch
            )
            if ckpt_path:
                policy_paths[label] = ckpt_path
                print(f"    Checkpoint : {ckpt_path}")
                print(f"    Epoch      : {ckpt_epoch}")
            else:
                print(f"    ⚠  No checkpoint found in {exp_dir}")
        else:
            print(f"    ⚠  Experiment directory not found for run {run_id}")

        print()

    if not all_results:
        print("No results collected.  Exiting.")
        sys.exit(1)

    # ── Print comparison table ───────────────────────────────────────────
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
            # Also print the eval command
            print(f"    Eval command:")
            print(f"      puffer eval puffer_drive --eval.wosac-realism-eval True "
                  f"--load-model-path {path}")
        else:
            print(f"  {label}: ⚠  No checkpoint found")
        print()

    # ── Optional CSV export ──────────────────────────────────────────────
    if args.csv:
        csv_path = args.csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(["Group", "Metric", "wandb_key"] + labels)
            # Data
            for group_name, group_metrics in METRIC_GROUPS.items():
                for key, display_name in group_metrics.items():
                    row_vals = []
                    for label in labels:
                        val = all_results[label].get(key)
                        row_vals.append(
                            val if val is not None else ""
                        )
                    writer.writerow([group_name, display_name, key] + row_vals)

            # Policy paths
            writer.writerow([])
            writer.writerow(["Policy Checkpoints"])
            for label in labels:
                writer.writerow([label, policy_paths.get(label, "N/A")])

        print(f"  ✓ Results saved to {csv_path}\n")

    print("Done.\n")


if __name__ == "__main__":
    main()
