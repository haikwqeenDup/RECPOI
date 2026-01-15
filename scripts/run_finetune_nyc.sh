#!/usr/bin/env bash
set -e

CACHE_PATH=${1:-cache/nyc_cache.pt}
PRETRAIN_CKPT=${2:-checkpoints/planC_pretrain_nyc.pt}

python -m src.finetune \
  --cache_path ${CACHE_PATH} \
  --save_dir checkpoints \
  --exp_name planC_finetune_nyc \
  --init_from ${PRETRAIN_CKPT} \
  --epochs 60 \
  --batch_size 256 \
  --lr 3e-4 \
  --weight_decay 1e-2 \
  --label_smoothing 0.05 \
  --lambda_cat 0.2 \
  --lambda_region 0.2 \
  --d_model 256 \
  --n_heads 8 \
  --n_layers 4 \
  --d_ff 1024 \
  --dropout 0.1 \
  --user_dim 128 \
  --pos2d_freq 8

python -m src.eval \
  --cache_path ${CACHE_PATH} \
  --ckpt_path checkpoints/planC_finetune_nyc.pt \
  --split test \
  --batch_size 256

echo "Finetune ckpt: checkpoints/planC_finetune_nyc.pt"
