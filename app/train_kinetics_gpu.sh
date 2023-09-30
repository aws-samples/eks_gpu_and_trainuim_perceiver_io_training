python3 -m torch.distributed.run --nproc_per_node=1 train_gpu.py \
  --config_file_path config/main.yaml \
  --dataset kinetics-small \
  --batches_per_epoch 1000 \
  --num_epochs 3 \
  --batch_size 8 \
  --learning_rate 1e-5
