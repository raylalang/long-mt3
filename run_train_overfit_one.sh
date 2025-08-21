export PYTORCH_ENABLE_MPS_FALLBACK=1
python3 train.py \
  data.manifest_path=manifests/manifest_20250818_030012.json \
  data.batch_size_per_device=1 \
  data.num_workers=0 \
  data.debug=true \
  data.overfit_one=true \
  data.segment_seconds=2.048 \
  model.d_model=64 \
  model.nhead=2 \
  model.dim_feedforward=128 \
  model.num_layers=2 \
  model.dropout=0.1 \
  train.learning_rate=5e-3 \
  train.max_epochs=100 \
  train.precision=32 \
  train.strategy=auto \
  train.accelerator=cpu \
  train.devices=1 \
  train.early_stop_patience=-1 \
  train.debug=true \
  train.tb_logger=True

# nohup bash ./run_train_overfit_one.sh > run_train_overfit_one.log 2>&1 &