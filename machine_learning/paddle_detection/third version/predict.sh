#!/bin/bash
set -euo pipefail

MODEL="runs/detect/train2/weights/best.pt"
OUTROOT="runs/detect/predict5"

for f in ../DataSet/[BC]0*.mp4; do
  stem="$(basename "$f" .mp4)"
  echo "Processing $stem ..."

  yolo detect predict \
    model="$MODEL" \
    source="$f" \
    project="$OUTROOT" \
    device=mps \
    show=False \
    name="$stem" \
    exist_ok=True

  # echo "Fixing orientation for $stem ..."
  # pred_path="$OUTROOT/$stem/$stem.mp4"

  # ffmpeg -y -loglevel error -i "$pred_path" -vf "vflip,hflip" "$OUTROOT/$stem/${stem}_fixed.mp4"
done

# yolo detect predict \
#   model="runs/detect/train2/weights/best.pt" \
#   source="../DataSet/video2.mp4" \
#   device=mps \
#   show=false
