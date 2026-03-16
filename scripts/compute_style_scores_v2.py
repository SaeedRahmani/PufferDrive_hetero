"""Compute extended style scores (z ∈ [-1,1]) including following-distance features.

This is a separate variant from compute_style_scores.py that adds two
leading-vehicle proximity features to the original four kinematic features:

  6-feature vector per agent:
    F1  mean_speed            (higher = more aggressive)
    F2  speed_p85             (higher = more aggressive)
    F3  accel_pos_p85         (higher = more aggressive)
    F4  brake_neg_p15         (higher = more aggressive)
    F5  inv_min_gap_distance  (smaller gap = more aggressive, inverted for ranking)
    F6  inv_min_time_headway  (smaller THW  = more aggressive, inverted for ranking)

  z = 2 * mean(percentile_ranks across 6 features) - 1

Leading vehicle detection:
  At each valid timestep, for each ego vehicle we search all other vehicles
  that lie within a forward cone (±30° of heading, within 100 m). The gap
  is the longitudinal projection minus half-lengths of both vehicles. Time
  headway = gap / ego_speed. We take the 5th-percentile across the
  trajectory as a robust near-minimum.

Binary output format (style_scores_v2.bin) — identical structure to v1:
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
import math
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VEHICLE, PEDESTRIAN, CYCLIST = 1, 2, 3
VALID_TYPES = {VEHICLE, PEDESTRIAN, CYCLIST}
MIN_VALID_STEPS = 30
COMPLETENESS_FRAC = 0.50
SPEED_THRESH = {VEHICLE: 0.5, PEDESTRIAN: 0.1, CYCLIST: 0.1}
SCENE_MIN_AGENTS = 5

# Leading-vehicle detection parameters
LEAD_MAX_DIST = 100.0       # metres — ignore vehicles further than this
LEAD_HALF_ANGLE = math.pi / 6  # ±30° forward cone
LEAD_MAX_LATERAL = 4.0      # metres — roughly one lane width each side
MIN_SPEED_FOR_THW = 0.5     # m/s — don't compute THW when nearly stopped
NUM_FEATURES = 6            # total feature count for this variant


# ---------------------------------------------------------------------------
# Binary reading (mirrors drive.h load_map_binary exactly)
# ---------------------------------------------------------------------------
def read_map(path):
    """Return (scenario_id_bytes, list[dict]) for one .bin map.

    Unlike v1 we also capture heading and vehicle length for gap computation.
    """
    agents = []
    with open(path, "rb") as f:
        scenario_id = f.read(16)
        _sdc_idx = struct.unpack("i", f.read(4))[0]
        n_predict = struct.unpack("i", f.read(4))[0]
        f.read(n_predict * 4)  # skip track indices
        num_objects = struct.unpack("i", f.read(4))[0]
        num_roads = struct.unpack("i", f.read(4))[0]
        num_entities = num_objects + num_roads

        for _ in range(num_entities):
            _sid = struct.unpack("i", f.read(4))[0]
            etype = struct.unpack("i", f.read(4))[0]
            eid = struct.unpack("i", f.read(4))[0]
            asize = struct.unpack("i", f.read(4))[0]

            traj_x = np.frombuffer(f.read(asize * 4), dtype=np.float32).copy()
            traj_y = np.frombuffer(f.read(asize * 4), dtype=np.float32).copy()
            _traj_z = f.read(asize * 4)

            if etype in VALID_TYPES:
                traj_vx = np.frombuffer(f.read(asize * 4), dtype=np.float32).copy()
                traj_vy = np.frombuffer(f.read(asize * 4), dtype=np.float32).copy()
                _vz = f.read(asize * 4)
                traj_heading = np.frombuffer(f.read(asize * 4), dtype=np.float32).copy()
                traj_valid = np.frombuffer(f.read(asize * 4), dtype=np.int32).copy()
                expert_accel = np.frombuffer(f.read(asize * 4), dtype=np.float32).copy()
                _steer = f.read(asize * 4)
            else:
                traj_vx = traj_vy = traj_heading = traj_valid = expert_accel = None

            # Scalar fields at end of each entity
            width = struct.unpack("f", f.read(4))[0]
            length = struct.unpack("f", f.read(4))[0]
            _height = struct.unpack("f", f.read(4))[0]
            _gx, _gy, _gz = struct.unpack("3f", f.read(12))
            _mark = struct.unpack("i", f.read(4))[0]

            if etype in VALID_TYPES:
                agents.append(dict(
                    etype=etype,
                    eid=eid,
                    asize=asize,
                    traj_x=traj_x,
                    traj_y=traj_y,
                    traj_vx=traj_vx,
                    traj_vy=traj_vy,
                    traj_heading=traj_heading,
                    traj_valid=traj_valid,
                    expert_accel=expert_accel,
                    length=length,
                    width=width,
                ))
    return scenario_id, agents


# ---------------------------------------------------------------------------
# Leading-vehicle gap computation (vectorised per scene per timestep)
# ---------------------------------------------------------------------------
def compute_leading_vehicle_features(agents):
    """For each agent, compute per-timestep gap and THW to closest leader.

    Returns two lists aligned with *agents*:
      gap_p5[i]  — 5th percentile of gap distances (or None if not enough data)
      thw_p5[i]  — 5th percentile of time headways (or None)
    """
    n = len(agents)
    if n == 0:
        return [], []

    # Find common trajectory length (all entities in a map share the same asize)
    T = agents[0]["asize"]

    # Pre-build per-timestep position / heading / valid / length arrays
    # Shape: (n, T)
    xs = np.stack([a["traj_x"] for a in agents])
    ys = np.stack([a["traj_y"] for a in agents])
    headings = np.stack([a["traj_heading"] for a in agents])
    valids = np.stack([a["traj_valid"] for a in agents]).astype(bool)
    vxs = np.stack([a["traj_vx"] for a in agents])
    vys = np.stack([a["traj_vy"] for a in agents])
    lengths = np.array([a["length"] for a in agents])  # (n,)
    etypes = np.array([a["etype"] for a in agents])

    # Only vehicles can have a "leading vehicle" relationship
    is_vehicle = etypes == VEHICLE

    # Per-agent accumulation
    gap_lists = [[] for _ in range(n)]
    thw_lists = [[] for _ in range(n)]

    cos_h = np.cos(headings)  # (n, T)
    sin_h = np.sin(headings)  # (n, T)
    speeds = np.sqrt(vxs ** 2 + vys ** 2)  # (n, T)

    for t in range(T):
        # Which agents are valid vehicles at this timestep?
        valid_t = valids[:, t] & is_vehicle  # (n,)
        idx = np.where(valid_t)[0]
        if len(idx) < 2:
            continue

        x_t = xs[idx, t]       # (m,)
        y_t = ys[idx, t]
        cos_t = cos_h[idx, t]
        sin_t = sin_h[idx, t]
        spd_t = speeds[idx, t]
        len_t = lengths[idx]

        # Pairwise relative positions: dx[i,j] = x_j - x_i
        dx = x_t[np.newaxis, :] - x_t[:, np.newaxis]  # (m, m)
        dy = y_t[np.newaxis, :] - y_t[:, np.newaxis]

        # Project into each ego's frame
        # longitudinal = dx * cos(heading_ego) + dy * sin(heading_ego)
        lon = dx * cos_t[:, np.newaxis] + dy * sin_t[:, np.newaxis]
        lat = -dx * sin_t[:, np.newaxis] + dy * cos_t[:, np.newaxis]

        # Gap = longitudinal distance minus half-lengths
        half_ego = len_t[:, np.newaxis] / 2.0
        half_other = len_t[np.newaxis, :] / 2.0
        gap = lon - half_ego - half_other

        # Mask: must be ahead (gap > 0), within lateral band, within range
        dist = np.sqrt(dx ** 2 + dy ** 2)
        mask = (
            (gap > 0)
            & (np.abs(lat) < LEAD_MAX_LATERAL)
            & (dist < LEAD_MAX_DIST)
        )
        np.fill_diagonal(mask, False)

        # For each ego, find minimum gap (vectorised)
        # Set non-candidate gaps to inf so argmin works
        gap_masked = np.where(mask, gap, np.inf)
        has_leader = mask.any(axis=1)  # (m,)
        min_gaps = gap_masked.min(axis=1)  # (m,)

        for local_i in np.where(has_leader)[0]:
            global_i = idx[local_i]
            mg = float(min_gaps[local_i])
            gap_lists[global_i].append(mg)
            if spd_t[local_i] > MIN_SPEED_FOR_THW:
                thw_lists[global_i].append(mg / spd_t[local_i])

    # Compute robust near-minimum (5th percentile)
    gap_p5 = []
    thw_p5 = []
    for i in range(n):
        if len(gap_lists[i]) >= 10:
            gap_p5.append(np.percentile(gap_lists[i], 5))
        else:
            gap_p5.append(None)
        if len(thw_lists[i]) >= 10:
            thw_p5.append(np.percentile(thw_lists[i], 5))
        else:
            thw_p5.append(None)

    return gap_p5, thw_p5


# ---------------------------------------------------------------------------
# Feature extraction (kinematic — same as v1)
# ---------------------------------------------------------------------------
def extract_kinematic_features(agent):
    """Return (F1, F2, F3, F4) or None if agent fails quality filters."""
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
    accel = accel[accel != -1.0]
    if len(accel) < 10:
        return None

    pos_accel = accel[accel > 0]
    f1 = mean_speed
    f2 = np.percentile(speed, 85)
    f3 = np.percentile(pos_accel, 85) if len(pos_accel) >= 5 else 0.0
    f4 = abs(np.percentile(accel, 15))
    return (f1, f2, f3, f4)


# ---------------------------------------------------------------------------
# Percentile ranking
# ---------------------------------------------------------------------------
def percentile_rank(value, sorted_values):
    """Fraction of values strictly less than value. *sorted_values* must be pre-sorted."""
    if len(sorted_values) <= 1:
        return 0.5
    return np.searchsorted(sorted_values, value, side="left") / len(sorted_values)


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------
def compute_scores_for_maps(all_maps):
    """
    Returns: list of (scenario_id_bytes, list[(agent_id, z)])
    """
    # ---- Step 1: extract per-agent features ----
    # feature_data[flat_idx] = (map_idx, agent_idx, etype, kin_feats_or_None)
    feature_data = []
    # gap_features[flat_idx] = (gap_p5, thw_p5) — may be None
    gap_features = []

    for mi, (_, agents) in enumerate(all_maps):
        gap_p5_list, thw_p5_list = compute_leading_vehicle_features(agents)
        for ai, ag in enumerate(agents):
            kin = extract_kinematic_features(ag)
            feature_data.append((mi, ai, ag["etype"], kin))
            gap_features.append((gap_p5_list[ai], thw_p5_list[ai]))
        if (mi + 1) % 500 == 0:
            print(f"  Extracted features: {mi+1}/{len(all_maps)}")

    # ---- Step 2: build feature pools ----
    # For features F1-F4 (kinematic) and F5-F6 (proximity, inverted)
    global_pools = defaultdict(lambda: [[] for _ in range(NUM_FEATURES)])
    scene_pools = defaultdict(lambda: defaultdict(lambda: [[] for _ in range(NUM_FEATURES)]))

    for flat_i, (mi, ai, etype, kin) in enumerate(feature_data):
        gp, tp = gap_features[flat_i]
        if kin is None:
            continue
        # Agent must be a vehicle with valid proximity data for the full 6-feature score
        has_proximity = (etype == VEHICLE and gp is not None and tp is not None)

        for fi in range(4):
            global_pools[etype][fi].append(kin[fi])
            scene_pools[mi][etype][fi].append(kin[fi])

        if has_proximity:
            # F5: smaller gap = more aggressive → store negative so higher = more aggressive
            neg_gap = -gp
            # F6: smaller THW = more aggressive → store negative
            neg_thw = -tp
            global_pools[etype][4].append(neg_gap)
            global_pools[etype][5].append(neg_thw)
            scene_pools[mi][etype][4].append(neg_gap)
            scene_pools[mi][etype][5].append(neg_thw)

    # Convert to sorted numpy arrays (sort once, reuse everywhere)
    for etype in global_pools:
        for fi in range(NUM_FEATURES):
            arr = np.array(global_pools[etype][fi]) if global_pools[etype][fi] else np.array([])
            arr.sort()
            global_pools[etype][fi] = arr

    # ---- Step 3: percentile ranking + z ----
    results = []
    fd_idx = 0
    for mi, (scen_id, agents) in enumerate(all_maps):
        map_scores = []
        for ai, ag in enumerate(agents):
            _, _, etype, kin = feature_data[fd_idx]
            gp, tp = gap_features[fd_idx]
            fd_idx += 1

            if kin is None:
                map_scores.append((ag["eid"], 0.0))
                continue

            has_proximity = (etype == VEHICLE and gp is not None and tp is not None)

            # Determine how many features to rank over
            if has_proximity:
                feat_vals = list(kin) + [-gp, -tp]
                n_feats = NUM_FEATURES
            else:
                feat_vals = list(kin)
                n_feats = 4

            ranks = []
            for fi in range(n_feats):
                sp = scene_pools[mi][etype][fi]
                if len(sp) >= SCENE_MIN_AGENTS:
                    pool = np.sort(sp)
                else:
                    pool = global_pools[etype][fi]  # already sorted
                if len(pool) == 0:
                    ranks.append(0.5)
                else:
                    ranks.append(percentile_rank(feat_vals[fi], pool))

            z = 2.0 * np.mean(ranks) - 1.0
            z = float(np.clip(z, -1.0, 1.0))
            map_scores.append((ag["eid"], z))
        results.append((scen_id, map_scores))
    return results


# ---------------------------------------------------------------------------
# Save binary (same format as v1)
# ---------------------------------------------------------------------------
def save_style_scores_bin(results, out_path):
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
    parser = argparse.ArgumentParser(description="Compute extended style scores (v2) with following-distance features")
    parser.add_argument("--data-dir", default=None,
                        help="Root containing training/ and validation/ subfolders")
    parser.add_argument("--out", default=None,
                        help="Output binary file path")
    parser.add_argument("--max-maps", type=int, default=0,
                        help="Max maps to process (0 = all)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(project_root, "pufferlib", "resources", "drive", "binaries")
    out_path = args.out or os.path.join(project_root, "pufferlib", "resources", "drive", "style_scores_v2.bin")

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
        if (i + 1) % 500 == 0:
            print(f"  Read {i+1}/{len(bin_files)}")
    print(f"Done reading {len(all_maps)} maps.")

    # Compute scores
    print("Computing extended style scores (v2)...")
    results = compute_scores_for_maps(all_maps)

    # Diagnostics
    print_diagnostics(results, all_maps)

    # Save
    save_style_scores_bin(results, out_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
