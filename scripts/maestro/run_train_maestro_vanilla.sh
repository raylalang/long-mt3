#!/usr/bin/env bash
set -euo pipefail
# nohup bash ./scripts/maestro/run_train_maestro_vanilla.sh > run_train_maestro_vanilla.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python3 train.py \
  data.manifest_path=manifests/manifest_20250827_181509.json \
  model.fusion.enabled=false \
  model.frontend.type=null \
  train.accelerator=gpu \
  train.devices=1 \
  train.precision=16-mixed \
  train.label_smoothing=0.1 \
  data.batch_size_per_device=16