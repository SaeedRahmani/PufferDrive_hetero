#!/usr/bin/env python3
"""
Collect Training Results
========================
Copies training artifacts from the three project directories into a single
``results/training_runs/`` folder in the current workspace, so that all
analysis can run from one place.

What gets collected for each paradigm (adversarial, guided reward, pure SP):
  • wandb metadata   – config.yaml, wandb-summary.json, wandb-metadata.json
  • wandb history    – full metric history exported to ``history.csv``
                       (via the wandb API, so you never need it again)
  • checkpoints      – all model_puffer_drive_*.pt files (or a subset)
  • training config  – pufferlib/config/ocean/drive.ini

Large files (output.log, .wandb binary) are **symlinked** by default to save
disk space.  Use ``--copy-all`` to do a full copy instead.

Output structure
----------------
    results/training_runs/
      adversarial/
        wandb/                    # wandb run metadata + exported history
          config.yaml
          wandb-summary.json
          wandb-metadata.json
          history.csv             # <-- full metric history, one row per step
        checkpoints/              # model_puffer_drive_*.pt
        drive.ini                 # training config snapshot
        run_info.json             # run ID, project dir, paradigm label, etc.
      guided_reward/
        ...
      pure_self_play/
        ...

Usage
-----
    # Default: collect everything, symlink large files
    python scripts/collect_training_results.py

    # Full copy (no symlinks)
    python scripts/collect_training_results.py --copy-all

    # Only copy specific checkpoint epochs (e.g. the one at ~5.35B steps)
    python scripts/collect_training_results.py --checkpoint-epochs 10201

    # Copy checkpoints at multiple epochs
    python scripts/collect_training_results.py --checkpoint-epochs 10201 5001 1001

    # Custom output directory
    python scripts/collect_training_results.py --output results/my_analysis
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path

# ── Project definitions ──────────────────────────────────────────────────────
# label → (directory name under --root, subfolder name in output)
PROJECTS: dict[str, tuple[str, str]] = {
    "Adversarial":    ("pufferdrive_adversarial",  "adversarial"),
    "Guided Reward":  ("pufferdrive_guidedreward", "guided_reward"),
    "Pure Self-Play": ("pufferdrive_puresp",       "pure_self_play"),
}


def find_latest_run(project_dir: str) -> tuple[str, str]:
    """Return (run_id, run_dir_path) for the latest wandb run."""
    wandb_dir = os.path.join(project_dir, "wandb")

    # Strategy 1: symlink
    latest_link = os.path.join(wandb_dir, "latest-run")
    if os.path.islink(latest_link):
        target = os.readlink(latest_link)
        run_id = target.rsplit("-", 1)[-1]
        run_dir = os.path.join(wandb_dir, target)
        if os.path.isdir(run_dir):
            return run_id, run_dir

    # Strategy 2: sort by name (timestamp prefix ensures latest sorts last)
    run_dirs = sorted(glob.glob(os.path.join(wandb_dir, "run-*")))
    if not run_dirs:
        raise FileNotFoundError(f"No wandb runs found in {wandb_dir}")
    run_dir = run_dirs[-1]
    run_id = os.path.basename(run_dir).rsplit("-", 1)[-1]
    return run_id, run_dir


def find_experiment_dir(project_dir: str, run_id: str) -> str | None:
    """Return the experiment checkpoint directory for a run."""
    exp_base = os.path.join(project_dir, "experiments")
    candidate = os.path.join(exp_base, f"puffer_drive_{run_id}")
    if os.path.isdir(candidate):
        return candidate
    for d in glob.glob(os.path.join(exp_base, f"*{run_id}*")):
        if os.path.isdir(d):
            return d
    return None


# ── Metric keys (must match compare_training_paradigms.py exactly) ────────────
# When exporting via the wandb API the same `keys` list must be passed so that
# the server-side downsampling returns the SAME rows the comparison script uses.
ALL_METRIC_KEYS: list[str] = [
    # WOSAC Realism
    "eval/wosac_realism_meta_score_mean", "eval/wosac_realism_meta_score_std",
    "eval/wosac_kinematic_metrics", "eval/wosac_interactive_metrics",
    "eval/wosac_map_based_metrics", "eval/wosac_likelihood_collision",
    "eval/wosac_likelihood_ttc", "eval/wosac_likelihood_dist_to_no",
    "eval/wosac_ade", "eval/wosac_min_ade", "eval/wosac_total_num_agents",
    "eval/num_collisions_sim", "eval/num_collisions_ref",
    # Evaluation
    "eval/sp_score", "eval/sp_collision_rate",
    "eval/hr_score", "eval/hr_collision_rate",
    "eval/human_log_prob",
    "eval/Δ_score", "eval/Δ_cr", "eval/Δ_or", "eval/Δ_comp",
    # Environment
    "environment/collision_rate", "environment/collisions_per_agent",
    "environment/offroad_rate", "environment/offroad_per_agent",
    "environment/completion_rate", "environment/goals_reached_this_episode",
    "environment/score", "environment/episode_return",
    "environment/dnf_rate", "environment/lane_alignment_rate",
    "environment/speed_at_goal", "environment/perturbed_agent_count",
    "environment/perturbed_collision_count",
    "environment/unperturbed_collision_count",
    # Losses
    "losses/policy_loss", "losses/value_loss", "losses/entropy",
    "losses/approx_kl", "losses/old_approx_kl", "losses/clipfrac",
    "losses/explained_variance", "losses/importance", "losses/human_nll",
    # Training
    "agent_steps", "epoch", "learning_rate", "SPS",
]


def export_wandb_history(
    entity: str, project: str, run_id: str, output_path: str
) -> bool:
    """Export wandb history to a CSV file via the API.

    IMPORTANT: We pass ``keys=ALL_METRIC_KEYS`` so that wandb's server-side
    downsampling returns the same rows that ``compare_training_paradigms.py``
    would receive.  Without the ``keys`` parameter the downsampling strategy
    differs and metric values at a given step won't match.
    """
    try:
        import wandb
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        keys_to_fetch = ["_step"] + ALL_METRIC_KEYS
        df = run.history(keys=keys_to_fetch, samples=10000, pandas=True)
        df.to_csv(output_path, index=False)
        print(f"      Exported {len(df)} rows to history.csv")
        return True
    except Exception as exc:
        print(f"      ⚠  Failed to export history: {exc}")
        return False


def copy_or_link(src: str, dst: str, use_symlink: bool) -> None:
    """Copy a file, or create a symlink to it."""
    if use_symlink:
        # Use absolute path for symlink target
        abs_src = os.path.abspath(src)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(abs_src, dst)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect training results from all paradigm directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root", type=str, default="/users/saeani/src",
        help="Root directory containing the project folders.",
    )
    parser.add_argument(
        "--output", type=str, default="results/training_runs",
        help="Output directory (relative to workspace or absolute).",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default="s-rahmani-tu-delft",
        help="Weights & Biases entity.",
    )
    parser.add_argument(
        "--wandb-project", type=str, default="pufferlib",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--copy-all", action="store_true",
        help="Full copy of large files instead of symlinking.",
    )
    parser.add_argument(
        "--checkpoint-epochs", type=int, nargs="*", default=None,
        help="Only copy checkpoints at these epochs. "
             "If omitted, copies ALL checkpoints.",
    )
    parser.add_argument(
        "--skip-history-export", action="store_true",
        help="Skip the wandb API history export (use if already exported).",
    )
    args = parser.parse_args()

    # Resolve output directory relative to the workspace
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(args.output):
        output_base = os.path.join(workspace_dir, args.output)
    else:
        output_base = args.output

    use_symlink = not args.copy_all

    print(f"\n{'═' * 70}")
    print(f"  COLLECT TRAINING RESULTS")
    print(f"  Output: {output_base}")
    print(f"  Mode: {'symlink large files' if use_symlink else 'full copy'}")
    print(f"{'═' * 70}\n")

    for label, (dirname, subfolder) in PROJECTS.items():
        project_dir = os.path.join(args.root, dirname)
        out_dir = os.path.join(output_base, subfolder)

        print(f"  ── {label} ──")
        print(f"    Source: {project_dir}")

        if not os.path.isdir(project_dir):
            print(f"    ⚠  Directory not found — skipping\n")
            continue

        # Find latest run
        try:
            run_id, run_dir = find_latest_run(project_dir)
        except FileNotFoundError as exc:
            print(f"    ⚠  {exc} — skipping\n")
            continue

        print(f"    Run ID: {run_id}")

        # ── Create output directories ────────────────────────────────────
        wandb_out = os.path.join(out_dir, "wandb")
        ckpt_out = os.path.join(out_dir, "checkpoints")
        os.makedirs(wandb_out, exist_ok=True)
        os.makedirs(ckpt_out, exist_ok=True)

        # ── Copy wandb metadata files ────────────────────────────────────
        files_dir = os.path.join(run_dir, "files")
        small_files = [
            "config.yaml",
            "wandb-summary.json",
            "wandb-metadata.json",
            "requirements.txt",
        ]
        print(f"    Copying wandb metadata...")
        for fname in small_files:
            src = os.path.join(files_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(wandb_out, fname))

        # Symlink or copy large files
        large_files = ["output.log"]
        for fname in large_files:
            src = os.path.join(files_dir, fname)
            if os.path.isfile(src):
                copy_or_link(src, os.path.join(wandb_out, fname), use_symlink)
                mode = "symlinked" if use_symlink else "copied"
                size_mb = os.path.getsize(src) / (1024 * 1024)
                print(f"      {fname} ({size_mb:.0f} MB) — {mode}")

        # Symlink or copy .wandb binary history file
        wandb_bin = glob.glob(os.path.join(run_dir, "run-*.wandb"))
        for wb in wandb_bin:
            dest = os.path.join(wandb_out, os.path.basename(wb))
            copy_or_link(wb, dest, use_symlink)
            size_mb = os.path.getsize(wb) / (1024 * 1024)
            mode = "symlinked" if use_symlink else "copied"
            print(f"      {os.path.basename(wb)} ({size_mb:.0f} MB) — {mode}")

        # ── Export wandb history to CSV ──────────────────────────────────
        history_csv = os.path.join(wandb_out, "history.csv")
        if not args.skip_history_export:
            print(f"    Exporting wandb history via API...")
            export_wandb_history(
                args.wandb_entity, args.wandb_project, run_id, history_csv
            )
        else:
            if os.path.isfile(history_csv):
                print(f"    history.csv already exists — skipping export")
            else:
                print(f"    ⚠  history.csv not found and --skip-history-export set")

        # ── Copy checkpoints ─────────────────────────────────────────────
        exp_dir = find_experiment_dir(project_dir, run_id)
        if exp_dir:
            ckpt_files = sorted(glob.glob(
                os.path.join(exp_dir, "model_puffer_drive_*.pt")
            ))

            if args.checkpoint_epochs is not None:
                # Filter to requested epochs only
                wanted = set(args.checkpoint_epochs)
                ckpt_files = [
                    f for f in ckpt_files
                    if int(re.search(r"_(\d+)\.pt$", f).group(1)) in wanted
                ]

            print(f"    Copying {len(ckpt_files)} checkpoint(s)...")
            for ckpt in ckpt_files:
                dst = os.path.join(ckpt_out, os.path.basename(ckpt))
                if not os.path.exists(dst):
                    shutil.copy2(ckpt, dst)
            print(f"      → {ckpt_out}/")

            # Also copy trainer_state.pt if it exists
            trainer_state = os.path.join(exp_dir, "trainer_state.pt")
            if os.path.isfile(trainer_state):
                shutil.copy2(
                    trainer_state,
                    os.path.join(ckpt_out, "trainer_state.pt")
                )
        else:
            print(f"    ⚠  No experiment directory found for run {run_id}")

        # ── Copy training config ─────────────────────────────────────────
        config_src = os.path.join(
            project_dir, "pufferlib", "config", "ocean", "drive.ini"
        )
        if os.path.isfile(config_src):
            shutil.copy2(config_src, os.path.join(out_dir, "drive.ini"))
            print(f"    Copied drive.ini")

        # ── Write run_info.json ──────────────────────────────────────────
        # Store metadata so the integrated script knows how to find things
        run_info = {
            "label": label,
            "subfolder": subfolder,
            "source_project_dir": project_dir,
            "run_id": run_id,
            "wandb_entity": args.wandb_entity,
            "wandb_project": args.wandb_project,
            "wandb_run_path": f"{args.wandb_entity}/{args.wandb_project}/{run_id}",
        }

        # Add summary info if available
        summary_path = os.path.join(wandb_out, "wandb-summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            run_info["total_steps"] = summary.get("agent_steps", summary.get("_step"))
            run_info["max_epoch"] = summary.get("epoch")

        with open(os.path.join(out_dir, "run_info.json"), "w") as f:
            json.dump(run_info, f, indent=2)
        print(f"    Wrote run_info.json")
        print()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"{'═' * 70}")
    print(f"  ✓ All results collected in: {output_base}")
    print(f"{'═' * 70}\n")

    # Show tree
    for subfolder in [v[1] for v in PROJECTS.values()]:
        sub = os.path.join(output_base, subfolder)
        if os.path.isdir(sub):
            print(f"  {subfolder}/")
            for root, dirs, files in os.walk(sub):
                level = root.replace(sub, "").count(os.sep)
                indent = "    " + "  " * level
                print(f"{indent}{os.path.basename(root)}/")
                file_indent = "    " + "  " * (level + 1)
                for f in sorted(files):
                    size = os.path.getsize(os.path.join(root, f))
                    if size > 1024 * 1024:
                        print(f"{file_indent}{f}  ({size / 1024 / 1024:.1f} MB)")
                    else:
                        print(f"{file_indent}{f}  ({size / 1024:.1f} KB)")
            print()


if __name__ == "__main__":
    main()
