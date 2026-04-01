#!/usr/bin/env bash
# ============================================================================
# run_behavior_analysis.sh
# Runs analyze_behavior.py for all 10 pufferdrive projects.
# Output → /users/saeani/src/guided_original/behavior_results/
# Log    → /tmp/behavior_analysis.log
# Usage:  bash run_behavior_analysis.sh
# ============================================================================
set -uo pipefail

SCRIPT="/users/saeani/src/guided_original/analyze_behavior.py"
OUT_DIR="/users/saeani/src/guided_original/behavior_results"
TRAIN_SC="0,1000,2000,3000,4000,5000,6000,7000,8000,8900"
VAL_SC="0,100,200,300,400,500,600,700,800,900"
EPISODE_STEPS=91
DT=0.1

mkdir -p "$OUT_DIR"

run_project() {
    local project_dir="$1"
    local venv_python="$2"
    local checkpoint="$3"
    local run_name="$4"

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo " Project : $run_name"
    echo " Dir     : $project_dir"
    echo " Ckpt    : $checkpoint"
    echo "════════════════════════════════════════════════════════"

    cd "$project_dir"
    "$venv_python" "$SCRIPT" \
        --checkpoint      "$checkpoint" \
        --run_name        "$run_name" \
        --output_dir      "$OUT_DIR" \
        --train_scenarios "$TRAIN_SC" \
        --val_scenarios   "$VAL_SC" \
        --episode_steps   "$EPISODE_STEPS" \
        --dt              "$DT"
}

# ── Project list ─────────────────────────────────────────────────────────────
run_project \
    /data/lynx/saeani/pufferdrive_guided_full \
    /data/lynx/saeani/pufferdrive_guided_full/.venv/bin/python \
    /data/lynx/saeani/pufferdrive_guided_full/experiments/puffer_drive_ylxme2w7.pt \
    guided_full

run_project \
    /data/lynx/saeani/pufferdrive_guided_full_dropout \
    /data/lynx/saeani/pufferdrive_guided_full_dropout/.venv/bin/python \
    /data/lynx/saeani/pufferdrive_guided_full_dropout/experiments/puffer_drive_txjswigu.pt \
    guided_full_dropout

run_project \
    /data/lynx/saeani/pufferdrive_guided_reward \
    /data/lynx/saeani/pufferdrive_guided_reward/.venv/bin/python \
    /data/lynx/saeani/pufferdrive_guided_reward/experiments/puffer_drive_80rf71mm.pt \
    guided_reward

run_project \
    /data/lynx/saeani/pufferdrive_hetero_v1 \
    /data/lynx/saeani/pufferdrive_hetero_v1/.venv/bin/python \
    /data/lynx/saeani/pufferdrive_hetero_v1/experiments/puffer_drive_83zq72gj.pt \
    hetero_v1

run_project \
    /data/lynx/saeani/pufferdrive_hetero_v2 \
    /data/lynx/saeani/pufferdrive_hetero_v2/.venv/bin/python \
    /data/lynx/saeani/pufferdrive_hetero_v2/experiments/puffer_drive_eodvv1ay.pt \
    hetero_v2

run_project \
    /data/lynx/saeani/pufferdrive_hetero_simple \
    /data/lynx/saeani/pufferdrive_hetero_simple/.venv/bin/python \
    /data/lynx/saeani/pufferdrive_hetero_simple/experiments/puffer_drive_txjswigu.pt \
    hetero_simple

run_project \
    /data/lynx/saeani/pufferdrive_vae_hybrid \
    /data/lynx/saeani/pufferdrive_vae_hybrid/.venv/bin/python \
    /data/lynx/saeani/pufferdrive_vae_hybrid/experiments/puffer_drive_wt6wuds6.pt \
    vae_hybrid

run_project \
    /data/lynx/saeani/pufferdrive_vae_hybrid_ll0 \
    /data/lynx/saeani/pufferdrive_vae_hybrid_ll0/.venv/bin/python \
    /data/lynx/saeani/pufferdrive_vae_hybrid_ll0/experiments/puffer_drive_j55uh86z.pt \
    vae_hybrid_ll0

run_project \
    /data/lynx/saeani/pufferdrive_causal_encoder \
    /data/lynx/saeani/pufferdrive_causal_encoder/.venv/bin/python \
    /data/lynx/saeani/pufferdrive_causal_encoder/experiments/puffer_drive_bngfvp94.pt \
    causal_encoder

run_project \
    /data/lynx/saeani/pufferdrive_causal_encoder_ll0 \
    /data/lynx/saeani/pufferdrive_causal_encoder_ll0/.venv/bin/python \
    /data/lynx/saeani/pufferdrive_causal_encoder_ll0/experiments/puffer_drive_i8z9igrx.pt \
    causal_encoder_ll0

echo ""
echo "════════════════════════════════════════════════════════"
echo " ALL BEHAVIOR ANALYSES COMPLETE"
echo " Results in: $OUT_DIR"
echo "════════════════════════════════════════════════════════"
