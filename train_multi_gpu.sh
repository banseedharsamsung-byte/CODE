#!/bin/bash
# Multi-GPU training script using accelerate
# Usage: ./train_multi_gpu.sh [num_gpus]

NUM_GPUS=${1:-4}  # Default to 4 GPUs if not specified

echo "Starting multi-GPU training with $NUM_GPUS GPUs..."

# Launch training with accelerate
accelerate launch --num_processes=$NUM_GPUS --mixed_precision=fp16 train.py

