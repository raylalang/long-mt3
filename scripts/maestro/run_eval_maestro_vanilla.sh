#!/usr/bin/env bash
set -euo pipefail
# nohup bash ./scripts/maestro/run_eval_maestro_vanilla.sh > run_eval_maestro_vanilla.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python3 eval.py \
  data.manifest_path=manifests/manifest_20250827_181509.json \
  eval.checkpoint=lightning_logs/version_11/checkpoints/last.ckpt \
  eval.hparams_yaml=lightning_logs/version_11/hparams.yaml \
  eval.accelerator=gpu \
  eval.devices=1