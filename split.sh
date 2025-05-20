#!/bin/bash

# paths
IMG_ROOT="/Users/virginiaceccatelli/Documents/CompVisionMorbius/hospital_environment"
ALL_IDS="all_ids.txt"
SPLITS_DIR="/Users/virginiaceccatelli/Documents/CompVisionMorbius/splits"

set -euo pipefail

mkdir -p "$SPLITS_DIR"

# 1) list all .jpg paths, strip prefix and extension
find "$IMG_ROOT" -type f -name '*.png' \
  | sed "s|^$IMG_ROOT/||;s|\.png$||" \
  > "$ALL_IDS"

# 2) count & compute 80%
total=$(wc -l < "$ALL_IDS")
train_count=$(( total * 80 / 100 ))

# 3) shuffle
gshuf "$ALL_IDS" > "${SPLITS_DIR}/all_ids_shuf.txt"

# 4) split
head -n "$train_count"    "${SPLITS_DIR}/all_ids_shuf.txt" > "${SPLITS_DIR}/train.txt"
tail -n +"$((train_count+1))" "${SPLITS_DIR}/all_ids_shuf.txt" > "${SPLITS_DIR}/val.txt"

# 5) cleanup
rm "${SPLITS_DIR}/all_ids_shuf.txt"
echo "Train size: $(wc -l < ${SPLITS_DIR}/train.txt), Val size: $(wc -l < ${SPLITS_DIR}/val.txt)"
