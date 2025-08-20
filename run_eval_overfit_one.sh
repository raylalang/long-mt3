python3 eval.py \
  eval.eval_one=true \
  eval.checkpoint=lightning_logs/version_0/checkpoints/last.ckpt \
  eval.hparams_yaml=lightning_logs/version_0/hparams.yaml \
  eval.accelerator=cpu \
  eval.devices=1 \
  eval.segment_batch_size=16 \
  eval.start_time=125 \
  eval.end_time=129 \
  data.manifest_path=manifests/manifest_20250818_030012.json \
  data.batch_size_per_device=1 \
  data.num_workers=0 \
  data.debug=true \
  data.overfit_one=true \
  data.segment_seconds=2.048