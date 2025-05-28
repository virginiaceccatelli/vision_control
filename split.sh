#!/bin/bash

# paths
IMG_ROOT="/Users/virginiaceccatelli/Documents/vision_control/CompVisionMorbius/hospital_environment"
ALL_IDS="all_ids.txt"
SPLITS_DIR="/Users/virginiaceccatelli/Documents/vision_control/CompVisionMorbius/splits"

set -euo pipefail

mkdir -p "$SPLITS_DIR"

# list all .jpg paths, strip prefix and extension
find "$IMG_ROOT" -type f -name '*.png' \
  | sed "s|^$IMG_ROOT/||;s|\.png$||" \
  > "$ALL_IDS"

# count & compute 80%
total=$(wc -l < "$ALL_IDS")
train_count=$(( total * 80 / 100 ))

# shuffle
gshuf "$ALL_IDS" > "${SPLITS_DIR}/all_ids_shuf.txt"

# split
head -n "$train_count"    "${SPLITS_DIR}/all_ids_shuf.txt" > "${SPLITS_DIR}/train.txt"
tail -n +"$((train_count+1))" "${SPLITS_DIR}/all_ids_shuf.txt" > "${SPLITS_DIR}/val.txt"

# cleanup
rm "${SPLITS_DIR}/all_ids_shuf.txt"
echo "Train size: $(wc -l < ${SPLITS_DIR}/train.txt), Val size: $(wc -l < ${SPLITS_DIR}/val.txt)"
