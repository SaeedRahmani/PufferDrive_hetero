"""
Adversarial Perturbation Evaluation Script
==========================================
Evaluates how well policies handle adversarial perturbation scenarios.

Supports two evaluation modes:
1. **Robustness sweep**: Evaluate a single policy across increasing perturbation
   fractions to measure degradation curves.
2. **Comparison**: Compare a baseline policy (trained without perturbation) against
   an adversarial policy (trained with perturbation) under identical conditions.

Metrics collected:
- Collision rate (overall, perturbed agents, unperturbed agents)
- Offroad rate
- Goal completion rate / score
- Episode return
- DNF (did-not-finish) rate

Usage:
    python examples/eval_adversarial.py --mode sweep --policy models/my_policy.pt
    python examples/eval_adversarial.py --mode compare \
        --baseline models/baseline.pt --adversarial models/adversarial.pt
"""

import argparse
import copy
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pufferlib.pufferl import load_config, load_env, load_policy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PERTURBATION_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def make_eval_config(
    env_name: str = "puffer_drive",
    perturbation_fraction: float = 0.0,
    use_perturbation_training: int = 1,
    perturb_aggressive_speed: float = 1.5,
    perturb_slow_speed: float = 0.5,
    perturb_heading_offset: float = 0.1,
    perturb_brake_duration: int = 10,
    num_maps: int = 100,
    map_dir: str = "resources/drive/binaries/validation",
    dynamics_model: str = "classic",
):
    """Build a config dict for adversarial evaluation."""
    config = load_config(env_name)

    # Evaluation-specific overrides
    config["env"]["num_maps"] = num_maps
    config["env"]["map_dir"] = map_dir
    config["env"]["dynamics_model"] = dynamics_model
    config["env"]["control_mode"] = "control_vehicles"
    config["env"]["init_mode"] = "create_all_valid"
    config["env"]["init_steps"] = 0
    config["env"]["save_data_to_disk"] = False
    config["env"]["prep_human_data"] = False
    config["vec"]["backend"] = "PufferEnv"

    # Perturbation settings
    config["env"]["use_perturbation_training"] = use_perturbation_training
    config["env"]["perturbation_fraction"] = perturbation_fraction
    config["env"]["perturb_aggressive_speed"] = perturb_aggressive_speed
    config["env"]["perturb_slow_speed"] = perturb_slow_speed
    config["env"]["perturb_heading_offset"] = perturb_heading_offset
    config["env"]["perturb_brake_duration"] = perturb_brake_duration

    # Training / policy settings for loading
    config["train"]["use_rnn"] = True

    return config


def collect_rollout_metrics(config, vecenv, policy, num_steps: int = 910):
    """Roll out a policy for num_steps and collect aggregate metrics.

    Returns a dict of averaged metrics over the rollout.
    """
    import torch

    policy.eval()
    device = next(policy.parameters()).device

    # Collect logs over the rollout
    all_logs = []
    obs = vecenv.observations
    lstm_state = None

    for step in range(num_steps):
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().to(device)
            if hasattr(policy, "encode_observations"):
                # PufferLib LSTM-based policy
                if lstm_state is None:
                    hidden = policy.encode_observations(obs_t)
                    lstm_state = policy.recurrent.initial_state(hidden.shape[0])
                    lstm_state = tuple(s.to(device) for s in lstm_state)
                hidden = policy.encode_observations(obs_t)
                hidden, lstm_state = policy.recurrent(hidden, lstm_state)
                actions = policy.decode_actions(hidden, deterministic=True)
            else:
                actions = policy(obs_t, deterministic=True)

            if isinstance(actions, tuple):
                actions = actions[0]
            actions = actions.cpu().numpy()

        vecenv.step(actions)
        obs = vecenv.observations

        # Collect any logs emitted by the env
        step_info = vecenv.info() if hasattr(vecenv, "info") else []
        for info_dict in step_info:
            if isinstance(info_dict, dict) and "n" in info_dict:
                all_logs.append(info_dict)

    return aggregate_logs(all_logs)


def aggregate_logs(logs):
    """Average a list of log dicts into a single summary dict."""
    if not logs:
        return {}

    keys = logs[0].keys()
    result = {}
    for k in keys:
        vals = [log[k] for log in logs if k in log and log[k] is not None]
        if vals:
            result[k] = float(np.mean(vals))
    return result


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------


def run_sweep(
    policy_path: str,
    fractions: list = None,
    num_maps: int = 100,
    num_steps: int = 910,
    dynamics_model: str = "classic",
    output_dir: str = "results/adversarial",
):
    """Evaluate a single policy across multiple perturbation fractions."""
    import torch

    fractions = fractions or PERTURBATION_FRACTIONS
    results = []

    for frac in fractions:
        print(f"\n{'='*60}")
        print(f"Evaluating perturbation_fraction = {frac:.2f}")
        print(f"{'='*60}")

        config = make_eval_config(
            perturbation_fraction=frac,
            use_perturbation_training=1 if frac > 0 else 0,
            num_maps=num_maps,
            dynamics_model=dynamics_model,
        )
        config["load_model_path"] = policy_path

        vecenv = load_env("puffer_drive", config)
        policy = load_policy(config, vecenv, "puffer_drive")
        policy.eval()

        metrics = collect_rollout_metrics(config, vecenv, policy, num_steps=num_steps)
        metrics["perturbation_fraction"] = frac
        metrics["policy"] = os.path.basename(policy_path)
        results.append(metrics)

        print(f"  collision_rate: {metrics.get('collision_rate', 'N/A')}")
        print(f"  score: {metrics.get('score', 'N/A')}")
        print(f"  perturbed_collision_count: {metrics.get('perturbed_collision_count', 'N/A')}")
        print(f"  unperturbed_collision_count: {metrics.get('unperturbed_collision_count', 'N/A')}")

    df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    plot_sweep(df, output_dir)
    return df


def run_comparison(
    baseline_path: str,
    adversarial_path: str,
    fractions: list = None,
    num_maps: int = 100,
    num_steps: int = 910,
    dynamics_model: str = "classic",
    output_dir: str = "results/adversarial",
):
    """Compare baseline vs adversarial-trained policy across perturbation levels."""
    import torch

    fractions = fractions or PERTURBATION_FRACTIONS
    results = []

    for label, path in [("baseline", baseline_path), ("adversarial", adversarial_path)]:
        for frac in fractions:
            print(f"\n{'='*60}")
            print(f"[{label}] perturbation_fraction = {frac:.2f}")
            print(f"{'='*60}")

            config = make_eval_config(
                perturbation_fraction=frac,
                use_perturbation_training=1 if frac > 0 else 0,
                num_maps=num_maps,
                dynamics_model=dynamics_model,
            )
            config["load_model_path"] = path

            vecenv = load_env("puffer_drive", config)
            policy = load_policy(config, vecenv, "puffer_drive")
            policy.eval()

            metrics = collect_rollout_metrics(config, vecenv, policy, num_steps=num_steps)
            metrics["perturbation_fraction"] = frac
            metrics["policy"] = label
            results.append(metrics)

            print(f"  collision_rate: {metrics.get('collision_rate', 'N/A')}")
            print(f"  score: {metrics.get('score', 'N/A')}")

    df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "comparison_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    plot_comparison(df, output_dir)
    return df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_sweep(df, output_dir):
    """Plot degradation curves for a single policy across perturbation fractions."""
    sns.set("notebook", font_scale=1.05)
    sns.set_style("ticks", rc={"figure.facecolor": "none", "axes.facecolor": "none"})

    metrics_to_plot = [
        ("collision_rate", "Collision Rate"),
        ("score", "Score"),
        ("offroad_rate", "Offroad Rate"),
        ("episode_return", "Episode Return"),
    ]

    available = [m for m in metrics_to_plot if m[0] in df.columns]
    n = len(available)
    if n == 0:
        print("No plottable metrics found. Skipping visualization.")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (col, title) in zip(axes, available):
        ax.plot(df["perturbation_fraction"], df[col], marker="o", linewidth=2)
        ax.set_xlabel("Perturbation Fraction")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3, linestyle="--")
        sns.despine(ax=ax)

    plt.tight_layout()
    path = os.path.join(output_dir, "sweep_degradation.png")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved sweep plot to {path}")
    plt.close()

    # Plot perturbed vs unperturbed collision counts
    if "perturbed_collision_count" in df.columns and "unperturbed_collision_count" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            df["perturbation_fraction"],
            df["perturbed_collision_count"],
            marker="s",
            label="Perturbed agents",
            linewidth=2,
        )
        ax.plot(
            df["perturbation_fraction"],
            df["unperturbed_collision_count"],
            marker="^",
            label="Unperturbed agents",
            linewidth=2,
        )
        ax.set_xlabel("Perturbation Fraction")
        ax.set_ylabel("Collision Count")
        ax.set_title("Collision Attribution")
        ax.legend()
        ax.grid(alpha=0.3, linestyle="--")
        sns.despine(ax=ax)
        plt.tight_layout()
        path = os.path.join(output_dir, "sweep_collision_attribution.png")
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved collision attribution plot to {path}")
        plt.close()


def plot_comparison(df, output_dir):
    """Plot side-by-side comparison of baseline vs adversarial policy."""
    sns.set("notebook", font_scale=1.05)
    sns.set_style("ticks", rc={"figure.facecolor": "none", "axes.facecolor": "none"})
    palette = sns.color_palette("Set2", n_colors=2)

    metrics_to_plot = [
        ("collision_rate", "Collision Rate"),
        ("score", "Score"),
        ("offroad_rate", "Offroad Rate"),
        ("episode_return", "Episode Return"),
    ]

    available = [m for m in metrics_to_plot if m[0] in df.columns]
    n = len(available)
    if n == 0:
        print("No plottable metrics found. Skipping visualization.")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (col, title) in zip(axes, available):
        for idx, policy_name in enumerate(df["policy"].unique()):
            subset = df[df["policy"] == policy_name]
            ax.plot(
                subset["perturbation_fraction"],
                subset[col],
                marker="o",
                linewidth=2,
                label=policy_name,
                color=palette[idx % len(palette)],
            )
        ax.set_xlabel("Perturbation Fraction")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3, linestyle="--")
        sns.despine(ax=ax)

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved comparison plot to {path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Adversarial Perturbation Evaluation")
    parser.add_argument(
        "--mode",
        choices=["sweep", "compare"],
        default="sweep",
        help="Evaluation mode: 'sweep' tests one policy across fractions, "
        "'compare' tests baseline vs adversarial.",
    )
    parser.add_argument("--policy", type=str, default=None, help="Path to policy checkpoint (sweep mode)")
    parser.add_argument("--baseline", type=str, default=None, help="Path to baseline policy (compare mode)")
    parser.add_argument("--adversarial", type=str, default=None, help="Path to adversarial policy (compare mode)")
    parser.add_argument("--num-maps", type=int, default=100, help="Number of map scenarios to evaluate")
    parser.add_argument("--num-steps", type=int, default=910, help="Steps per evaluation rollout")
    parser.add_argument("--dynamics", type=str, default="classic", choices=["classic", "jerk"])
    parser.add_argument("--output-dir", type=str, default="results/adversarial")
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=None,
        help="Custom perturbation fractions to evaluate",
    )

    args = parser.parse_args()

    matplotlib.use("Agg")  # Non-interactive backend for server use

    if args.mode == "sweep":
        if args.policy is None:
            parser.error("--policy is required for sweep mode")
        run_sweep(
            policy_path=args.policy,
            fractions=args.fractions,
            num_maps=args.num_maps,
            num_steps=args.num_steps,
            dynamics_model=args.dynamics,
            output_dir=args.output_dir,
        )
    elif args.mode == "compare":
        if args.baseline is None or args.adversarial is None:
            parser.error("--baseline and --adversarial are required for compare mode")
        run_comparison(
            baseline_path=args.baseline,
            adversarial_path=args.adversarial,
            fractions=args.fractions,
            num_maps=args.num_maps,
            num_steps=args.num_steps,
            dynamics_model=args.dynamics,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
