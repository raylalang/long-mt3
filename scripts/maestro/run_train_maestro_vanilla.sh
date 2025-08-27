#!/usr/bin/env bash
set -euo pipefail
# nohup bash ./scripts/overfit_one/run_train_maestro_vanilla.sh > run_train_maestro_vanilla.log 2>&1 &

python3 train.py \
  data.manifest_path=manifests/manifest_20250818_030012.json