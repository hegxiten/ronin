#!/bin/bash
set -e
echo "================================================================================"
echo "RoNIN Quick Hardware PoC"
echo "================================================================================"
source .venv/bin/activate
echo "a001_1
a001_3
a002_1
a002_2
a003_1" > lists/list_train_tiny.txt
echo ""
echo "[1/2] Quick training test (3 epochs, 5 sequences)..."
python source/ronin_resnet.py \
  --mode train \
  --train_list lists/list_train_tiny.txt \
  --root_dir datasets/ronin \
  --out_dir models/from_scratch/poc_quick \
  --epochs 3 \
  --batch_size 4 \
  --lr 0.001
echo ""
echo "[2/2] GPU Status..."
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
echo ""
echo "================================================================================"
echo "✓ PoC Complete - CUDA Training: WORKING"
echo "================================================================================"
