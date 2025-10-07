#!/usr/bin/env bash
set -euo pipefail

# 2) 記得確保 data.yaml 指到正確路徑（相對於 data.yaml）
#   train: ./train/images
#   val:   ./val/images
#   nc: 1
#   names: ['paddle']

echo "[i] using data.yaml:"
grep -E '^(train|val|nc|names):' data.yaml || true

# check if there exist ./val
if [ ! -d "val/images" ] || [ ! -d "val/labels" ]; then
  echo "[!] 沒看到 val/ 資料夾，先跑 make_val_split.py 再來訓練。"
  exit 1
fi

yolo detect train data=./data.yaml model=yolo11n.pt epochs=100 imgsz=640

# find the newest runs/detect/train*/ directory
LATEST_DIR=$(ls -td runs/detect/train*/ 2>/dev/null | head -1)
if [ -z "${LATEST_DIR:-}" ]; then
  echo "[x] didn't find runs/detect/train*/ directory, training process might filed." >&2
  exit 1
fi
BEST="${LATEST_DIR%/}/weights/best.pt"

VIDEO="${VIDEO:-../DataSet/source1.mp4}"
if [ -f "$BEST" ]; then
  echo "[i] predicting on $VIDEO using $BEST"
  yolo detect predict model="$BEST" source="$VIDEO" save=True
  echo "[OK] results -> runs/detect/predict*/"
else
  echo "[x] 找不到 $BEST，訓練是不是失敗了？" >&2
  exit 1
fi
