#!/usr/bin/env bash
# End-to-end Reproduction Script for DHA-Rec
set -e

echo "=== DHA-Rec Reproduction Pipeline ==="
echo "[1/3] Downloading datasets..."
bash scripts/download_data.sh ./data

echo "[2/3] Training DHA-Rec on MovieLens-1M..."
python3 scripts/train.py --dataset MovieLens-1M --epochs 5

echo "[3/3] Reproduction complete."
