#!/usr/bin/env bash
set -euo pipefail
# nohup bash ./scripts/overfit_one/11_segment_seconds_sweep.sh > 11_segment_seconds_sweep.log 2>&1 &

ENTRY="python3 train.py"
COMMON=(
  data.manifest_path=manifests/manifest_20250825_163452.json
  data.batch_size_per_device=1
  data.num_workers=0
  data.debug=true
  data.overfit_one=true
  model.d_model=64
  model.nhead=2
  model.dim_feedforward=128
  model.num_layers=2
  model.dropout=0.1
  model.frontend.type=unet
  model.fusion.enabled=true
  model.fusion.attn_kind=vanilla
  model.fusion.beats_per_bar=4
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

for S in 1.024 2.048 4.096; do
  echo "[segment_seconds=${S}]"
  $ENTRY "${COMMON[@]}" data.segment_seconds="${S}"
done
