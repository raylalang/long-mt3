#!/usr/bin/env bash
set -euo pipefail
# nohup bash ./scripts/overfit_one/10_baseline_unet_fusion.sh > 10_baseline_unet_fusion.log 2>&1 &

ENTRY="python3 train.py"
COMMON=(
  data.manifest_path=manifests/manifest_20250825_163452.json
  data.batch_size_per_device=1
  data.num_workers=0
  data.debug=true
  data.overfit_one=true
  data.segment_seconds=2.048
  model.d_model=64
  model.nhead=2
  model.dim_feedforward=128
  model.num_layers=2
  model.dropout=0.1
  model.frontend.type=null
  model.fusion.enabled=false
  train.learning_rate=5e-3
  train.max_epochs=3
  train.precision=32
  train.strategy=auto
  train.accelerator=cpu
  train.devices=1
  train.early_stop_patience=-1
  train.debug=true
  train.tb_logger=True
)

echo "[1/3] Baseline (no UNet, no fusion)"
$ENTRY "${COMMON[@]}"

echo "[2/3] UNet front-end (no fusion)"
$ENTRY "${COMMON[@]}" \
  model.frontend.type=unet \
  model.fusion.enabled=false

echo "[3/3] UNet + Fusion (vanilla; beat+bar)"
$ENTRY "${COMMON[@]}" \
  model.frontend.type=unet \
  model.fusion.enabled=true \
  model.fusion.attn_kind=vanilla \
  model.fusion.beats_per_bar=4
