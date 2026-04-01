"""
Standalone scenario visualization script.
Plots road edges + ground truth agent trajectories for map_000.bin.
Centers coordinates to fix the "tiny plot" issue from world-frame coordinates.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Run from guided_original directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from pufferlib.ocean.drive.drive import Drive

MAP_DIR = "resources/drive/binaries"
OUTPUT_PNG = "scenario_visualization.png"


def main():
    print("Initializing Drive environment...")
    env = Drive(
        map_dir=MAP_DIR,
        num_maps=1,
        num_agents=128,          # smaller = faster init
        episode_length=91,
        init_steps=0,
        resample_frequency=91,
        use_guided_autonomy=0,
        use_guidance_observations=0,
        prep_human_data=False,
        save_data_to_disk=False,
    )

    print("Resetting environment...")
    env.reset()

    print("Fetching trajectories and road edges...")
    traj = env.get_ground_truth_trajectories()
    road = env.get_road_edge_polylines()

    # --- Extract coordinates ---
    # traj['x'] shape: (num_agents, 1, T) after the reshape in get_ground_truth_trajectories
    # Squeeze to (num_agents, T)
    traj_x = traj["x"].squeeze(1)   # (num_agents, T)
    traj_y = traj["y"].squeeze(1)
    traj_valid = traj["valid"].squeeze(1)  # (num_agents, T)

    road_x = road["x"]   # flattened
    road_y = road["y"]

    # --- Center on scenario centroid to fix tiny-plot issue ---
    all_x = np.concatenate([traj_x[traj_valid == 1], road_x])
    all_y = np.concatenate([traj_y[traj_valid == 1], road_y])

    if len(all_x) == 0:
        print("ERROR: No valid data found!")
        return

    cx, cy = all_x.mean(), all_y.mean()
    print(f"World-frame centroid: ({cx:.1f}, {cy:.1f}) m")

    traj_x = traj_x - cx
    traj_y = traj_y - cy
    road_x = road_x - cx
    road_y = road_y - cy

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 12))

    # Road edges
    lengths = road["lengths"]
    pt_idx = 0
    for seg_len in lengths:
        sx = road_x[pt_idx : pt_idx + seg_len]
        sy = road_y[pt_idx : pt_idx + seg_len]
        ax.plot(sx, sy, color="black", linewidth=0.8, alpha=0.6)
        pt_idx += seg_len

    print(f"Plotted {len(lengths)} road edge segments")

    # Agent trajectories
    num_agents = traj_x.shape[0]
    T = traj_x.shape[1]
    colormap = cm.get_cmap("tab20", num_agents)

    agents_plotted = 0
    for i in range(num_agents):
        valid_mask = traj_valid[i] == 1
        if valid_mask.sum() < 2:
            continue
        xs = traj_x[i, valid_mask]
        ys = traj_y[i, valid_mask]
        ts = np.where(valid_mask)[0]

        # Color line segments by time
        for t in range(len(xs) - 1):
            frac = ts[t] / max(T - 1, 1)
            ax.plot(xs[t:t+2], ys[t:t+2], color=cm.viridis(frac), linewidth=1.2, alpha=0.7)

        # Mark start
        ax.plot(xs[0], ys[0], "go", markersize=4, alpha=0.8)
        # Mark end
        ax.plot(xs[-1], ys[-1], "rs", markersize=4, alpha=0.8)
        agents_plotted += 1

    print(f"Plotted {agents_plotted} agent trajectories")

    # Colorbar for time
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, T * 0.1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Time (s)")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black", linewidth=1.5, label="Road edges"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green",
               markersize=8, label="Agent start"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="red",
               markersize=8, label="Agent end"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    ax.set_aspect("equal")
    ax.set_xlabel("X (m, centered)", fontsize=12)
    ax.set_ylabel("Y (m, centered)", fontsize=12)
    ax.set_title(f"map_000.bin — {agents_plotted} agents, {len(lengths)} road segments\n"
                 f"World centroid: ({cx:.0f}, {cy:.0f}) m", fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
