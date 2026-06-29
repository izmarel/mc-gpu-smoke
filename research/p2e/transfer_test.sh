#!/usr/bin/env bash
# (C) TRANSFER TEST — does memory from one world help in a DIFFERENT regenerated world?
# (the thing Plan2Explore structurally cannot do).
#
#   World A (seed 1): explore; SAVE the memory graph + the trained world-model weights.
#   World B carry (seed 2 = regenerated world): a FRESH run that LOADS world-A's weights
#                 (so latents match the saved memory keys) AND carries world-A memory;
#                 measure the in-world-B prediction gain from memory (ablation).
#   World B wipe  (seed 2): same fresh run + world-A weights, but FRESH memory (control).
#
# Both B arms carry world-A's WEIGHTS (required: the memory keys are that model's latents),
# fresh optimizer/ratio so world B trains normally. The ONLY difference is world-A memory.
#   TRANSFER = carry's in-world-B memory-gain  >  wipe's  (extra help that came from world A).
#
# Usage: bash research/p2e/transfer_test.sh [steps_A] [steps_B]
set -euo pipefail
cd "$(dirname "$0")/../.."

STEPS_A="${1:-300}"
STEPS_B="${2:-300}"

export MEM_AUG=1 DRIVE=lp MEM_ABLATE=1 MEM_ABLATE_EVERY=4 SCOREBOARD=1
export PYTHONPATH="research/p2e/world:research/p2e/retrieval:research/p2e/drive:research/p2e/monitor:."
export SCOREBOARD_DIR="$(pwd)/logs/transfer"
mkdir -p "$SCOREBOARD_DIR"
STORE_A="$SCOREBOARD_DIR/storeA.pkl"
MODEL_A="$SCOREBOARD_DIR/modelA.ckpt"
rm -f "$SCOREBOARD_DIR"/ablation_*.json

common=(exp=p2e_dv3_exploration env=crafter
  'env.wrapper._target_=crafter_tvs.CrafterTVWrapper'
  fabric.accelerator=cpu env.num_envs=1 env.sync_env=True
  'algo.cnn_keys.encoder=[rgb]' 'algo.cnn_keys.decoder=[rgb]'
  'algo.mlp_keys.encoder=[memory]' 'algo.mlp_keys.decoder=[]'
  algo.per_rank_batch_size=4 algo.per_rank_sequence_length=8
  algo.learning_starts=64 buffer.size=2000 algo.run_test=False
  metric.log_level=1 metric.log_every=100
  checkpoint.save_last=False model_manager.disabled=True)

echo "########## WORLD A (seed 1): explore -> save memory + world-model weights ##########"
ARM=worldA MEM_SAVE="$STORE_A" MODEL_SAVE="$MODEL_A" \
  sheeprl "${common[@]}" seed=1 algo.total_steps="$STEPS_A"
[ -f "$MODEL_A" ] || { echo "NO world-model saved — abort"; exit 1; }

echo "########## WORLD B carry (seed 2): fresh run + world-A weights + CARRY memory ##########"
ARM=transfer_carry MODEL_LOAD="$MODEL_A" MEM_LOAD="$STORE_A" \
  sheeprl "${common[@]}" seed=2 algo.total_steps="$STEPS_B"

echo "########## WORLD B wipe (seed 2): fresh run + world-A weights + FRESH memory ##########"
ARM=transfer_wipe MODEL_LOAD="$MODEL_A" \
  sheeprl "${common[@]}" seed=2 algo.total_steps="$STEPS_B"

echo "########## TRANSFER RESULT ##########"
python - <<'PY'
import json, os
d = os.environ["SCOREBOARD_DIR"]
def load(a):
    p = os.path.join(d, f"ablation_{a}.json")
    return json.load(open(p)) if os.path.exists(p) else None
c, w = load("transfer_carry"), load("transfer_wipe")
print("carry:", c)
print("wipe :", w)
if c and w:
    print()
    print(f"  world-B mem-gain  CARRY={c['mem_gain']:+.3f}   WIPE={w['mem_gain']:+.3f}")
    print(f"  TRANSFER (carry - wipe) = {c['mem_gain'] - w['mem_gain']:+.3f}")
    print("  >0  -> memory carried from world A lowered prediction error in world B (transfer).")
    print("  ~0  -> no transfer (worlds too dissimilar, or memory keys stale).")
else:
    print("  (missing ablation json — a B arm did not complete)")
PY
