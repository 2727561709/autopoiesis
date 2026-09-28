#!/bin/sh
# 第三轮放大实验（b 次尝试）：密度匹配的退火终点
# 16x16 的 2 份食物密度(0.78%)远低于 12x12 的 2 份(1.4%)，导致 a 次存活崩溃；
# 本次 food_end=4 (4/256≈1.6%) 与 12x12 终局密度匹配，训练加长到 800 轮。
set -e
cd /app
python -m homeo_rl.train --updates 800 --grid-size 16 --food-anneal \
  --food-start 12 --food-end 4 --visit-trace --out runs/scale16b \
  > /app/runs/scale16b_train.log 2>&1
python -m homeo_rl.evaluate --ckpt runs/scale16b/checkpoint.pt \
  --transfer --out runs/scale16b > /app/runs/scale16b_eval.log 2>&1
echo "SCALE16B_DONE" >> /app/runs/scale16b_train.log
