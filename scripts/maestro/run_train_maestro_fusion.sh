#!/usr/bin/env bash
set -euo pipefail
# nohup bash ./scripts/maestro/run_train_maestro_fusion.sh > run_train_maestro_fusion.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python3 train.py \
  data.manifest_path=manifests/manifest_20250827_181509.json \
  model.fusion.enabled=true \
  model.frontend.type=unet \
  model.fusion.beats_per_bar=4 \
  train.accelerator=gpu \
  train.devices=1 \
  train.precision=32 \
  +train.label_smoothing=0.1 \
  +train.gradient_clip_val=1.0 \
  data.batch_size_per_device=16