#!/usr/bin/env bash
# run_all_rollouts.sh
# Runs rollout_template.py for all 10 projects using each project's own venv.
# Fixed scenarios (same across all projects for fair comparison):
#   Train : map indices 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 8900
#   Val   : map indices 0,  100,  200,  300,  400,  500,  600,  700,  800,  900

set -e

TEMPLATE="/users/saeani/src/guided_original/rollout_template.py"
OUTPUT_DIR="/users/saeani/src/guided_original/rollouts"
TRAIN_SC="0,1000,2000,3000,4000,5000,6000,7000,8000,8900"
VAL_SC="0,100,200,300,400,500,600,700,800,900"
BASE="/data/lynx/saeani"

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  Project definitions: PROJECT_DIR  VENV_PYTHON  CHECKPOINT  RUN_NAME   │
# └─────────────────────────────────────────────────────────────────────────┘

declare -A PROJECTS
#                  proj_dir                         venv_python                                                         checkpoint                                                    run_name
PROJECTS[01]="${BASE}/pufferdrive_guided_full      | ${BASE}/pufferdrive_guided_full/.venv/bin/python       | ${BASE}/pufferdrive_guided_full/experiments/puffer_drive_ylxme2w7.pt        | guided_full"
PROJECTS[02]="${BASE}/pufferdrive_guided_full_dropout | ${BASE}/pufferdrive_guided_full_dropout/.venv/bin/python | ${BASE}/pufferdrive_guided_full_dropout/experiments/puffer_drive_txjswigu.pt  | guided_full_dropout"
PROJECTS[03]="${BASE}/pufferdrive_guided_reward    | ${BASE}/pufferdrive_guided_reward/.venv/bin/python     | ${BASE}/pufferdrive_guided_reward/experiments/puffer_drive_80rf71mm.pt       | guided_reward"
PROJECTS[04]="${BASE}/pufferdrive_hetero_v1        | ${BASE}/pufferdrive_hetero_v1/.venv/bin/python         | ${BASE}/pufferdrive_hetero_v1/experiments/puffer_drive_83zq72gj.pt           | hetero_v1"
PROJECTS[05]="${BASE}/pufferdrive_hetero_v2        | ${BASE}/pufferdrive_hetero_v2/.venv/bin/python         | ${BASE}/pufferdrive_hetero_v2/experiments/puffer_drive_eodvv1ay.pt           | hetero_v2"
PROJECTS[06]="${BASE}/pufferdrive_causal_encoder   | ${BASE}/pufferdrive_causal_encoder/.venv/bin/python    | ${BASE}/pufferdrive_causal_encoder/experiments/puffer_drive_bngfvp94.pt      | causal_encoder"
PROJECTS[07]="${BASE}/pufferdrive_causal_encoder_ll0 | ${BASE}/pufferdrive_causal_encoder_ll0/.venv/bin/python | ${BASE}/pufferdrive_causal_encoder_ll0/experiments/puffer_drive_i8z9igrx.pt | causal_enc_ll0"
PROJECTS[08]="${BASE}/pufferdrive_hetero_simple    | ${BASE}/pufferdrive_hetero_simple/.venv/bin/python     | ${BASE}/pufferdrive_hetero_simple/experiments/puffer_drive_txjswigu.pt       | hetero_simple"
PROJECTS[09]="${BASE}/pufferdrive_vae_hybrid       | ${BASE}/pufferdrive_vae_hybrid/.venv/bin/python        | ${BASE}/pufferdrive_vae_hybrid/experiments/puffer_drive_wt6wuds6.pt          | vae_hybrid"
PROJECTS[10]="${BASE}/pufferdrive_vae_hybrid_ll0   | ${BASE}/pufferdrive_vae_hybrid_ll0/.venv/bin/python    | ${BASE}/pufferdrive_vae_hybrid_ll0/experiments/puffer_drive_j55uh86z.pt      | vae_hybrid_ll0"

mkdir -p "$OUTPUT_DIR"

for key in $(echo "${!PROJECTS[@]}" | tr ' ' '\n' | sort); do
    entry="${PROJECTS[$key]}"
    # Parse pipe-separated values (strip surrounding whitespace)
    proj_dir=$(echo "$entry" | cut -d'|' -f1 | xargs)
    python=$(echo "$entry" | cut -d'|' -f2 | xargs)
    ckpt=$(echo "$entry" | cut -d'|' -f3 | xargs)
    name=$(echo "$entry" | cut -d'|' -f4 | xargs)

    echo ""
    echo "████████████████████████████████████████████████████████████████████"
    echo "  Project : $name"
    echo "  Dir     : $proj_dir"
    echo "  Python  : $python"
    echo "  Ckpt    : $ckpt"
    echo "████████████████████████████████████████████████████████████████████"

    (
        cd "$proj_dir"
        "$python" "$TEMPLATE" \
            --checkpoint "$ckpt" \
            --run_name   "$name" \
            --output_dir "$OUTPUT_DIR" \
            --train_scenarios "$TRAIN_SC" \
            --val_scenarios   "$VAL_SC"
    )
done

echo ""
echo "ALL DONE — rollouts saved to $OUTPUT_DIR"
