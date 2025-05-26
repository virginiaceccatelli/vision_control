#!/usr/bin/env bash


OUT_FILE=metrics2.csv # can be changed for new visualizations 

infile=${1:-/dev/stdin}
outfile=${2:-"$OUT_FILE"}

echo "epoch,train_loss,val_loss,val_iou" > "$outfile"

grep '^\[Epoch ' "$infile" | sed -E \
  -e 's/^\[Epoch[[:space:]]*([0-9]+)\/[0-9]+\][[:space:]]*train_loss=([0-9.]+)[[:space:]]*val_loss=([0-9.]+)[[:space:]]*val_iou=([0-9.]+).*/\1,\2,\3,\4/' \
  >> "$outfile"
