#!/usr/bin/env bash
# Run ONE experiment arm and append its result to the shared scoreboard, so
# scoreboard.report() can compare arms against the random (chance) baseline.
#
# Usage:  bash research/p2e/run_arms.sh <arm> <total_steps> [extra hydra overrides...]
#   arms:
#     full    = MEM_AUG=1 DRIVE=lp        (memory + learning-progress drive — the design)
#     nomem   = MEM_AUG=0 DRIVE=lp        (drive only — isolates memory's effect)
#     stock   = MEM_AUG=0 DRIVE=ensemble  (standard Plan2Explore curiosity baseline)
#     random  = random actions            (CHANCE baseline / the ruler — never trains)
#
# Behavioural lie-detector (time-on-TV etc.) is written to logs/scoreboard/.
# After running the arms you care about:  python research/p2e/monitor/scoreboard.py logs/scoreboard
set -euo pipefail
cd "$(dirname "$0")/../.."

ARM="${1:?arm: full|nomem|stock|random}"
STEPS="${2:?total_steps}"
SEED="${3:-42}"
shift 3 2>/dev/null || shift $# || true


export SCOREBOARD=1
export ARM="$ARM"
export SCOREBOARD_DIR="$(pwd)/logs/scoreboard"
export PYTHONPATH="research/p2e/world:research/p2e/retrieval:research/p2e/drive:research/p2e/monitor:."
# bound both memory stores (FIFO) so a long run can't OOM. Override by exporting these.
export MEM_DRIVE_MAX="${MEM_DRIVE_MAX:-100000}"
export MEM_INJECT_MAX="${MEM_INJECT_MAX:-100000}"

# per-arm config (match on the base arm name; results are tagged with the seed)
BASE="$ARM"
MEM=0; DRIVE=ensemble; MLP_ENC='[]'; LEARN_STARTS=64
case "$BASE" in
  full)   MEM=1; DRIVE=lp;       MLP_ENC='[memory]' ;;
  nomem)  MEM=0; DRIVE=lp;       MLP_ENC='[]' ;;
  stock)  MEM=0; DRIVE=ensemble; MLP_ENC='[]' ;;
  random) MEM=0; DRIVE=ensemble; MLP_ENC='[]'; LEARN_STARTS=100000000 ;;  # never trains -> all random
  *) echo "unknown arm: $BASE"; exit 1 ;;
esac
export MEM_AUG="$MEM"
export DRIVE="$DRIVE"
export ARM="${BASE}_s${SEED}"   # tag scoreboard/ablation json per (arm, seed)

# CPU defaults for validation; for the GPU run override ACCEL=gpu and pass real
# config via extra args, e.g.: ACCEL=cuda bash run_arms.sh full 200000 1 \
#   algo.per_rank_batch_size=16 algo.per_rank_sequence_length=64 buffer.size=1000000
ACCEL="${ACCEL:-cpu}"
BATCH="${BATCH:-4}"; SEQ="${SEQ:-8}"; BUF="${BUF:-2000}"

echo ">>> arm=$ARM  seed=$SEED  accel=$ACCEL  MEM_AUG=$MEM  DRIVE=$DRIVE  mlp=$MLP_ENC  steps=$STEPS"
exec sheeprl exp=p2e_dv3_exploration env=crafter \
  'env.wrapper._target_=crafter_tvs.CrafterTVWrapper' \
  "fabric.accelerator=$ACCEL" seed="$SEED" \
  env.num_envs=1 env.sync_env=True \
  'algo.cnn_keys.encoder=[rgb]' 'algo.cnn_keys.decoder=[rgb]' \
  "algo.mlp_keys.encoder=$MLP_ENC" 'algo.mlp_keys.decoder=[]' \
  algo.total_steps="$STEPS" algo.learning_starts="$LEARN_STARTS" \
  algo.per_rank_batch_size="$BATCH" algo.per_rank_sequence_length="$SEQ" \
  buffer.size="$BUF" \
  algo.run_test=False \
  metric.log_level=1 metric.log_every=100 \
  checkpoint.save_last=False model_manager.disabled=True \
  "$@"
