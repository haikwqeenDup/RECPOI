# Plan C: Spatio‑Temporal Masked Pretrain -> Next‑POI Fine‑tune (工程实现)

这个工程实现的是你提出的 **方案C**：

1. **Masked Pretrain（BERT4Rec 风格）**：对轨迹序列做 BERT 掩码预测（MLM），同时预测辅助目标（类别/多尺度区域）。
2. **Fine‑tune Next‑POI**：用 causal Transformer（只看历史）预测下一跳 POI，同时保留辅助任务。

> 你之前的基线（扩散 + gate + InfoNCE + SID）是另一条路线；这里是“预训练范式”路线，用来突破监督学习的上限。

---

## 目录结构

```
planC_poi_st_pretrain/
  requirements.txt
  src/
    data/
      preprocess.py
      features.py
      dataset_pretrain.py
      dataset_finetune.py
      collate.py
    models/
      model.py
    utils/
      seed.py
      metrics.py
      checkpoint.py
    pretrain.py
    finetune.py
    eval.py
  scripts/
    run_pretrain_nyc.sh
    run_finetune_nyc.sh
  cache/
  checkpoints/
```

---

## 数据格式要求

`NYC_train.csv / NYC_val.csv / NYC_test.csv` 至少包含列（大小写不敏感）：

- `trajectory_id`
- `user_id`
- `POI_id`
- `UTC_time` （或 timestamp/datetime）
- 可选：`timezone`（分钟偏移，例如 -240）
- 可选：`time_period`（0~1 之间的小数，表示一天内位置；若存在优先用它算 TOD）
- 可选：`latitude, longitude, POI_catid`（如果 poi_info 缺失某些 POI 的属性，会用这里补全）

`poi_info.csv`（可选但推荐）至少包含：

- `poi_id`
- `latitude`
- `longitude`
- 可选：`poi_catid`

---

## 安装

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 1) 预处理（生成 cache，一次即可）

```bash
python -m src.data.preprocess \
  --train_path data/NYC/NYC_train.csv \
  --val_path data/NYC/NYC_val.csv \
  --test_path data/NYC/NYC_test.csv \
  --poi_info_path data/NYC/poi_info.csv \
  --cache_path cache/nyc_cache.pt \
  --max_len 128 \
  --num_tod_bins 48 \
  --region_scales_m 700,1200,3000
```

---

## 2) Masked Pretrain

```bash
python -m src.pretrain \
  --cache_path cache/nyc_cache.pt \
  --save_dir checkpoints \
  --exp_name planC_pretrain_nyc \
  --epochs 30 \
  --batch_size 256 \
  --lr 3e-4 \
  --weight_decay 1e-2 \
  --mask_prob 0.25
```

输出：
- `checkpoints/planC_pretrain_nyc.pt`

---

## 3) Fine‑tune Next‑POI（加载预训练权重）

```bash
python -m src.finetune \
  --cache_path cache/nyc_cache.pt \
  --save_dir checkpoints \
  --exp_name planC_finetune_nyc \
  --init_from checkpoints/planC_pretrain_nyc.pt \
  --epochs 60 \
  --batch_size 256 \
  --lr 3e-4 \
  --weight_decay 1e-2 \
  --label_smoothing 0.05
```

输出：
- `checkpoints/planC_finetune_nyc.pt`

---

## 4) 测试集评估

```bash
python -m src.eval \
  --cache_path cache/nyc_cache.pt \
  --ckpt_path checkpoints/planC_finetune_nyc.pt \
  --split test
```

会输出：
- Acc@1/5/10/20
- MRR

---

## 一键脚本（可选）

```bash
bash scripts/run_pretrain_nyc.sh data/NYC cache/nyc_cache.pt
bash scripts/run_finetune_nyc.sh cache/nyc_cache.pt checkpoints/planC_pretrain_nyc.pt
```

---

## 重要可调超参（建议你优先扫这些）

- `--num_tod_bins`：48（30min）/ 96（15min）常能提升
- `--mask_prob`：0.15~0.30
- `--max_len`：128 -> 256（更强但更慢）
- `--n_layers` / `--d_model`：容量不足会欠拟合；太大容易过拟合（NYC中等规模一般 4层/256 很稳）
