#!/usr/bin/env bash
set -e

DATA_DIR=${1:-data/NYC}
CACHE_PATH=${2:-cache/nyc_cache.pt}

python -m src.data.preprocess \
  --train_path ${DATA_DIR}/NYC_train.csv \
  --val_path ${DATA_DIR}/NYC_val.csv \
  --test_path ${DATA_DIR}/NYC_test.csv \
  --poi_info_path ${DATA_DIR}/poi_info.csv \
  --cache_path ${CACHE_PATH} \
  --max_len 128 \
  --num_tod_bins 48 \
  --region_scales_m 700,1200,3000

python -m src.pretrain \
  --cache_path ${CACHE_PATH} \
  --save_dir checkpoints \
  --exp_name planC_pretrain_nyc \
  --epochs 30 \
  --batch_size 256 \
  --lr 3e-4 \
  --weight_decay 1e-2 \
  --mask_prob 0.25 \
  --lambda_cat 0.2 \
  --lambda_region 0.2 \
  --d_model 256 \
  --n_heads 8 \
  --n_layers 4 \
  --d_ff 1024 \
  --dropout 0.1 \
  --user_dim 128 \
  --pos2d_freq 8

echo "Pretrain ckpt: checkpoints/planC_pretrain_nyc.pt"
