"""Compute offline style scores (z ∈ [-1,1]) for all agents in Waymo binary maps.

Pipeline:
  1. Read all binary maps (training + validation).
  2. Per agent: extract 4 raw features (mean_speed, speed_p85, accel_pos_p85, brake_neg_p15).
  3. Within-scene, within-type percentile ranking (fallback to global when <5 same-type agents).
  4. z = 2 * mean(percentile_ranks) - 1.
  5. Save to resources/drive/style_scores.bin for C consumption.

Binary output format (style_scores.bin):
  int32  num_maps
  Per map:
    char[16]  scenario_id
    int32     num_agents_in_map
    Per agent:
      int32   agent_id
      float32 style_score
"""

import os
import struct
import argparse
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VEHICLE, PEDESTRIAN, CYCLIST = 1, 2, 3
VALID_TYPES = {VEHICLE, PEDESTRIAN, CYCLIST}
MIN_VALID_STEPS = 30        # ≥ 30 valid timesteps
COMPLETENESS_FRAC = 0.50    # ≥ 50 % of trajectory length must be valid
SPEED_THRESH = {             # minimum mean speed to be considered "active"
    VEHICLE: 0.5,
    PEDESTRIAN: 0.1,
    CYCLIST: 0.1,
}
SCENE_MIN_AGENTS = 5         # min same-type active agents for scene-level ranking


# ---------------------------------------------------------------------------
# Binary reading (mirrors drive.h load_map_binary)
# ---------------------------------------------------------------------------
def read_map(path):
    """Return (scenario_id_bytes, list[dict]) for one .bin map."""
    agents = []
    with open(path, "rb") as f:
        scenario_id = f.read(16)
        sdc_idx = struct.unpack("i", f.read(4))[0]
        n_predict = struct.unpack("i", f.read(4))[0]
        f.read(n_predict * 4)  # skip track indices
        num_objects = struct.unpack("i", f.read(4))[0]
        num_roads = struct.unpack("i", f.read(4))[0]
        num_entities = num_objects + num_roads

        for _ in range(num_entities):
            sid = struct.unpack("i", f.read(4))[0]
            etype = struct.unpack("i", f.read(4))[0]
            eid = struct.unpack("i", f.read(4))[0]
            asize = struct.unpack("i", f.read(4))[0]

            traj_x = np.frombuffer(f.read(asize * 4), dtype=np.float32).copy()
            traj_y = np.frombuffer(f.read(asize * 4), dtype=np.float32).copy()
            _traj_z = f.read(asize * 4)  # skip z

            if etype in VALID_TYPES:
                traj_vx = np.frombuffer(f.read(asize * 4), dtype=np.float32).copy()
                traj_vy = np.frombuffer(f.read(asize * 4), dtype=np.float32).copy()
                _vz = f.read(asize * 4)
                _heading = f.read(asize * 4)
                traj_valid = np.frombuffer(f.read(asize * 4), dtype=np.int32).copy()
                expert_accel = np.frombuffer(f.read(asize * 4), dtype=np.float32).copy()
                _steer = f.read(asize * 4)
            else:
                traj_vx = traj_vy = traj_valid = expert_accel = None

            # Scalar fields at end of each entity
            _w, _l, _h = struct.unpack("3f", f.read(12))
            _gx, _gy, _gz = struct.unpack("3f", f.read(12))
            _mark = struct.unpack("i", f.read(4))[0]

            if etype in VALID_TYPES:
                agents.append(dict(
                    etype=etype,
                    eid=eid,
                    asize=asize,
                    traj_vx=traj_vx,
                    traj_vy=traj_vy,
                    traj_valid=traj_valid,
                    expert_accel=expert_accel,
                ))
    return scenario_id, agents


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(agent):
    """Return (F1, F2, F3, F4) or None if agent doesn't pass filters."""
    valid = agent["traj_valid"].astype(bool)
    n_valid = valid.sum()
    if n_valid < MIN_VALID_STEPS:
        return None
    if n_valid < COMPLETENESS_FRAC * agent["asize"]:
        return None

    speed = np.sqrt(agent["traj_vx"][valid] ** 2 + agent["traj_vy"][valid] ** 2)
    mean_speed = speed.mean()
    if mean_speed < SPEED_THRESH.get(agent["etype"], 0.5):
        return None

    accel = agent["expert_accel"][valid]
    # expert_accel can be -1 for invalid — filter those
    accel = accel[accel != -1.0]
    if len(accel) < 10:
        return None

    pos_accel = accel[accel > 0]
    f1 = mean_speed
    f2 = np.percentile(speed, 85)
    f3 = np.percentile(pos_accel, 85) if len(pos_accel) >= 5 else 0.0
    f4 = abs(np.percentile(accel, 15))  # higher = harder braking
    return (f1, f2, f3, f4)


# ---------------------------------------------------------------------------
# Percentile ranking
# ---------------------------------------------------------------------------
def percentile_rank(value, values):
    """Fraction of values strictly less than value."""
    if len(values) <= 1:
        return 0.5
    return np.searchsorted(np.sort(values), value, side="left") / len(values)


def compute_scores_for_maps(all_maps):
    """
    all_maps: list of (scenario_id_bytes, agents_list)
    Returns: list of (scenario_id_bytes, list[(agent_id, z)])  for ALL agents (active or not)
    """
    # Step 1 — Extract features globally
    # For each (map_idx, agent_idx_in_list): features or None
    feature_data = []  # (map_idx, agent_idx, etype, features_tuple)
    for mi, (_, agents) in enumerate(all_maps):
        for ai, ag in enumerate(agents):
            feats = extract_features(ag)
            feature_data.append((mi, ai, ag["etype"], feats))

    # Step 2 — Build global feature pools per type (for fallback)
    global_features = defaultdict(lambda: [[] for _ in range(4)])
    for mi, ai, etype, feats in feature_data:
        if feats is not None:
            for fi in range(4):
                global_features[etype][fi].append(feats[fi])

    for etype in global_features:
        for fi in range(4):
            global_features[etype][fi] = np.array(global_features[etype][fi])

    # Step 3 — Build scene-level feature pools per type
    scene_features = defaultdict(lambda: defaultdict(lambda: [[] for _ in range(4)]))
    for mi, ai, etype, feats in feature_data:
        if feats is not None:
            for fi in range(4):
                scene_features[mi][etype][fi].append(feats[fi])

    # Step 4 — Percentile ranking + z computation
    results = []
    fd_idx = 0
    for mi, (scen_id, agents) in enumerate(all_maps):
        map_scores = []
        for ai, ag in enumerate(agents):
            _, _, etype, feats = feature_data[fd_idx]
            fd_idx += 1
            if feats is None:
                map_scores.append((ag["eid"], 0.0))
                continue

            ranks = []
            for fi in range(4):
                scene_pool = scene_features[mi][etype][fi]
                if len(scene_pool) >= SCENE_MIN_AGENTS:
                    pool = np.array(scene_pool)
                else:
                    pool = global_features[etype][fi]
                ranks.append(percentile_rank(feats[fi], pool))

            z = 2.0 * np.mean(ranks) - 1.0
            z = float(np.clip(z, -1.0, 1.0))
            map_scores.append((ag["eid"], z))
        results.append((scen_id, map_scores))
    return results


# ---------------------------------------------------------------------------
# Save binary
# ---------------------------------------------------------------------------
def save_style_scores_bin(results, out_path):
    """
    Binary format:
      int32  num_maps
      Per map:
        char[16]  scenario_id
        int32     num_agents
        Per agent:
          int32   agent_id
          float32 style_score
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("i", len(results)))
        for scen_id, map_scores in results:
            f.write(scen_id)  # 16 bytes
            f.write(struct.pack("i", len(map_scores)))
            for aid, z in map_scores:
                f.write(struct.pack("i", aid))
                f.write(struct.pack("f", z))
    print(f"Saved {out_path}  ({len(results)} maps)")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def print_diagnostics(results, all_maps):
    all_z = defaultdict(list)
    for (_, agents), (_, scores) in zip(all_maps, results):
        for ag, (aid, z) in zip(agents, scores):
            all_z[ag["etype"]].append(z)

    type_name = {VEHICLE: "Vehicle", PEDESTRIAN: "Pedestrian", CYCLIST: "Cyclist"}
    for etype in sorted(all_z):
        arr = np.array(all_z[etype])
        active = arr[arr != 0.0]
        print(f"\n{type_name.get(etype, etype)}:")
        print(f"  Total agents: {len(arr)}")
        print(f"  Active (z!=0): {len(active)}")
        if len(active) > 0:
            print(f"  z  mean={active.mean():.3f}  std={active.std():.3f}")
            print(f"     p5={np.percentile(active,5):.3f}  p25={np.percentile(active,25):.3f}  "
                  f"p50={np.percentile(active,50):.3f}  p75={np.percentile(active,75):.3f}  "
                  f"p95={np.percentile(active,95):.3f}")
            print(f"     min={active.min():.3f}  max={active.max():.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute style scores for Waymo binaries")
    parser.add_argument("--data-dir", default=None,
                        help="Root containing training/ and validation/ subfolders")
    parser.add_argument("--out", default=None,
                        help="Output binary file path")
    parser.add_argument("--max-maps", type=int, default=0,
                        help="Max maps to process (0 = all)")
    args = parser.parse_args()

    # Default paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(project_root, "resources", "drive", "binaries")
    out_path = args.out or os.path.join(project_root, "resources", "drive", "style_scores.bin")

    # Collect binary files
    bin_files = []
    for split in ["training", "validation"]:
        split_dir = os.path.join(data_dir, split)
        if os.path.isdir(split_dir):
            files = sorted([os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith(".bin")])
            print(f"  {split}: {len(files)} maps")
            bin_files.extend(files)
    print(f"Total binary files: {len(bin_files)}")

    if args.max_maps > 0:
        bin_files = bin_files[: args.max_maps]
        print(f"Processing first {len(bin_files)} maps only")

    # Read all maps
    print("Reading maps...")
    all_maps = []
    for i, path in enumerate(bin_files):
        scen_id, agents = read_map(path)
        all_maps.append((scen_id, agents))
        if (i + 1) % 1000 == 0:
            print(f"  Read {i+1}/{len(bin_files)}")
    print(f"Done reading {len(all_maps)} maps.")

    # Compute scores
    print("Computing style scores...")
    results = compute_scores_for_maps(all_maps)

    # Diagnostics
    print_diagnostics(results, all_maps)

    # Save
    save_style_scores_bin(results, out_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
