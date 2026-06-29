#!/usr/bin/env bash
# Combined CPU dry-run: memory-augmentation (MEM_AUG=1) + learning-progress drive
# (DRIVE=lp) through the real p2e_dv3 pipeline on seeded Crafter with TV noise tiles.
# Proves the whole wiring trains end-to-end before any GPU spend. Few-hundred steps.
set -euo pipefail
cd "$(dirname "$0")/../.."


export MEM_AUG=1
export MEM_K=4
export DRIVE=lp
export PYTHONPATH="research/p2e/world:research/p2e/retrieval:research/p2e/drive:."

exec sheeprl exp=p2e_dv3_exploration env=crafter \
  'env.wrapper._target_=crafter_tvs.CrafterTVWrapper' \
  fabric.accelerator=cpu \
  env.num_envs=1 env.sync_env=True \
  'algo.cnn_keys.encoder=[rgb]' 'algo.cnn_keys.decoder=[rgb]' \
  'algo.mlp_keys.encoder=[memory]' 'algo.mlp_keys.decoder=[]' \
  algo.total_steps=400 algo.learning_starts=64 \
  algo.per_rank_batch_size=4 algo.per_rank_sequence_length=8 \
  buffer.size=2000 \
  algo.run_test=False \
  metric.log_level=1 metric.log_every=50 \
  checkpoint.save_last=False model_manager.disabled=True \
  "$@"
