#!/bin/sh
# 第三轮放大实验：16x16 + 访问痕迹记忆 + 食物退火 -> 评测（含迁移探针）
# Round-3 scale-up: 16x16 + visit-trace memory + food annealing -> eval (with transfer)
set -e
cd /app
python -m homeo_rl.train --updates 500 --grid-size 16 --food-anneal \
  --food-start 12 --food-end 2 --visit-trace --out runs/scale16 \
  > /app/runs/scale16_train.log 2>&1
python -m homeo_rl.evaluate --ckpt runs/scale16/checkpoint.pt \
  --transfer --out runs/scale16 > /app/runs/scale16_eval.log 2>&1
echo "SCALE16_DONE" >> /app/runs/scale16_train.log
